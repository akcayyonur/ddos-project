# src/ryu_ddos_mitigator.py
#
# NSL-KDD tabanlı ML modelini kullanarak SDN'de DDoS benzeri trafiği
# "trafik şekillendirme" ile yumuşak şekilde bastıran Ryu uygulaması.
#
# Davranış (DFNet-lite tarzı):
#  - ARP ve diğer non-IP trafiği: sadece controller üzerinden PacketOut ile iletilir
#    (FLOW YAZMIYORUZ, böylece IPv4 trafiği mutlaka ML/queue logic'e gelir).
#  - IPv4 trafiği:
#       * Flow istatistiği çıkarılır.
#       * ML modeli ile benign/attack tahmini yapılır.
#       * Benign flow  -> yüksek öncelikli kuyruk (queue 0)
#       * Attack flow  -> düşük öncelikli kuyruk (queue 1)
#  - Böylece saldırı trafiği tamamen DROP edilmez ancak bandwidth anlamında boğulur.

import sys
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, icmp

# -------------------------------------------------------------
# Proje kökünü sys.path'e ekle ki src.config vb. import edilebilsin
# -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, META_PATH  # noqa: E402


class DDoSMitigator(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DDoSMitigator, self).__init__(*args, **kwargs)

        # ----------------- ML modeli yükle -----------------
        model_path = Path(MODEL_PATH)
        meta_path = Path(META_PATH)

        if not model_path.exists():
            raise RuntimeError(f"ML model not found: {model_path}")
        if not meta_path.exists():
            raise RuntimeError(f"Meta file not found: {meta_path}")

        self.model = joblib.load(model_path)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        # Eğitim sırasında kullanılan gerçek feature sırası
        self.feature_order = meta.get("feature_order", [])

        # Meta dosyasında hesaplanan threshold
        meta_thr = float(meta.get("threshold", 0.5))
        # Runtime'da meta ile aynı kullanıyoruz (istersen oynayabilirsin).
        self.threshold = meta_thr

        # Flow istatistikleri (ML için)
        # key: (dpid, src_ip, dst_ip, proto, src_port, dst_port)
        self.flow_stats = {}

        # Learning switch L2 tablosu: {dpid: {mac: port}}
        self.mac_to_port = {}

        # ML devreye girmeden önce akış başına minimum paket sayısı
        self.min_pkts_for_ml = 3

        # Ek heuristik: bu kadar paketi geçen uzun flow'u saldırı say
        self.attack_pkt_threshold = 50

        self.logger.info("=== ML MODEL LOADED ===")
        self.logger.info("model path   : %s", model_path)
        self.logger.info("meta path    : %s", meta_path)
        self.logger.info("feature_order: %s", self.feature_order)
        self.logger.info("meta threshold : %.4f", meta_thr)
        self.logger.info("runtime thr   : %.4f", self.threshold)
        self.logger.info("min_pkts_for_ml      : %d", self.min_pkts_for_ml)
        self.logger.info("attack_pkt_threshold : %d", self.attack_pkt_threshold)

    # -------------------------------------------------------------
    # Yardımcı: queue-aware flow ekleme (sadece IPv4 için kullanıyoruz)
    # -------------------------------------------------------------
    def add_flow_with_queue(
        self,
        datapath,
        priority,
        match,
        out_port,
        queue_id,
        idle_timeout=60,
        hard_timeout=0,
    ):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser

        # Queue ID, switch tarafında OVS QoS/queue konfigürasyonu ile eşleşmeli.
        actions = [
            parser.OFPActionSetQueue(queue_id),
            parser.OFPActionOutput(out_port),
        ]

        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
            instructions=inst,
        )
        datapath.send_msg(mod)

    # -------------------------------------------------------------
    # Switch bağlanınca: table-miss kuralı
    # -------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser

        self.logger.info("Switch connected: dpid=%s", datapath.id)

        # Table-miss: eşleşmeyen paketleri controller'a gönder
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER,
                                          ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=0,
            match=match,
            instructions=inst,
        )
        datapath.send_msg(mod)

    # -------------------------------------------------------------
    # ML için istatistik ve feature çıkarımı
    # -------------------------------------------------------------
    def _update_flow_stats(self, dpid, src, dst, proto, src_port, dst_port, pkt_len):
        key = (dpid, src, dst, proto, src_port, dst_port)
        now = time.time()

        st = self.flow_stats.get(key)
        if st is None:
            st = {
                "first_seen": now,
                "last_seen": now,
                "packet_count": 0,
                "byte_count": 0,
                "error_count": 0,
            }
            self.flow_stats[key] = st

        st["last_seen"] = now
        st["packet_count"] += 1
        st["byte_count"] += pkt_len

        return key, st

    def _build_feature_vector(self, stats):
        """
        flow_stats kaydından, model_meta.json'daki feature_order'a göre
        tek satırlık bir DataFrame üretir.
        """
        duration = max(0.0, stats["last_seen"] - stats["first_seen"])
        src_bytes = float(stats["byte_count"])
        dst_bytes = 0.0  # Mininet senaryosunda elimizde yok; 0 kabul ediyoruz.
        count = float(stats["packet_count"])
        srv_count = count
        serror_rate = float(stats["error_count"]) / max(1.0, count)

        bytes_ratio = (src_bytes + 1.0) / (dst_bytes + 1.0)
        count_ratio = (srv_count + 1.0) / (count + 1.0)
        log_src_bytes = np.log1p(src_bytes)
        log_dst_bytes = np.log1p(dst_bytes)

        feat_map = {
            "duration": duration,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "count": count,
            "srv_count": srv_count,
            "serror_rate": serror_rate,
            "bytes_ratio": bytes_ratio,
            "count_ratio": count_ratio,
            "log_src_bytes": log_src_bytes,
            "log_dst_bytes": log_dst_bytes,
        }

        row = [float(feat_map.get(name, 0.0)) for name in self.feature_order]
        X = pd.DataFrame([row], columns=self.feature_order)
        return X

    def _predict_attack(self, X, pkt_count: int):
        """
        Model + heuristik:
          - predict_proba >= threshold  => attack
          - veya packet_count >= attack_pkt_threshold => attack
        """
        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(X)[0, 1])
            is_attack = (proba >= self.threshold) or (
                pkt_count >= self.attack_pkt_threshold
            )
            y = 1 if is_attack else 0
            return y, proba
        else:
            y_raw = int(self.model.predict(X)[0])
            proba = None
            is_attack = (y_raw == 1) or (pkt_count >= self.attack_pkt_threshold)
            y = 1 if is_attack else 0
            return y, proba

    # -------------------------------------------------------------
    # PacketIn handler
    # -------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        dpid = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        # LLDP'yi görmezden gel
        if eth.ethertype == 0x88cc:
            return

        src_mac = eth.src
        dst_mac = eth.dst

        # Learning switch tablosunu güncelle (L2 için hala tutuyoruz)
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port

        # Hedef MAC biliniyorsa o porta, yoksa flood
        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            out_port = ofp.OFPP_FLOOD

        # -------------------------------------------------
        # 1) Non-IP (özellikle ARP) trafiği:
        #    Sadece PacketOut ile forward ediyoruz, FLOW YAZMIYORUZ.
        #    Böylece IPv4 trafiği her zaman ML/queue logic'e uğrar.
        # -------------------------------------------------
        ip = pkt.get_protocol(ipv4.ipv4)
        if ip is None:
            actions = [parser.OFPActionOutput(out_port)]
            out = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None,
            )
            datapath.send_msg(out)
            return

        # -------------------------------------------------
        # 2) IPv4 trafik: ML ile analiz et (DFNet-lite shaping)
        # -------------------------------------------------
        src_ip = ip.src
        dst_ip = ip.dst
        proto = ip.proto

        src_port = 0
        dst_port = 0

        t = pkt.get_protocol(tcp.tcp)
        u = pkt.get_protocol(udp.udp)
        ic = pkt.get_protocol(icmp.icmp)

        if t is not None:
            src_port = t.src_port
            dst_port = t.dst_port
        elif u is not None:
            src_port = u.src_port
            dst_port = u.dst_port
        elif ic is not None:
            # ICMP için port yok
            pass

        pkt_len = len(msg.data)

        key, stats = self._update_flow_stats(
            dpid, src_ip, dst_ip, proto, src_port, dst_port, pkt_len,
        )

        # === WARM-UP: Çok az paket varsa ML'e sokma ===
        if stats["packet_count"] < self.min_pkts_for_ml:
            self.logger.info(
                "[WARMUP] %s -> %s pkt_count=%d (ML devre dışı)",
                src_ip,
                dst_ip,
                stats["packet_count"],
            )
            actions = [parser.OFPActionOutput(out_port)]
            out = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None,
            )
            datapath.send_msg(out)
            return
        # === WARM-UP SONU ===

        # Yeterli paket geldiyse ML'e gönder
        X = self._build_feature_vector(stats)
        y_pred, proba = self._predict_attack(X, stats["packet_count"])

        # Benign -> high priority queue (0)
        # Attack -> low priority queue (1)
        if y_pred == 1:
            queue_id = 1
            self.logger.warning(
                "[SHAPE-ATTACK] low-priority queue=1: %s:%s -> %s:%s "
                "(pkt_count=%d, proba=%.3f, key=%s)",
                src_ip,
                src_port,
                dst_ip,
                dst_port,
                stats["packet_count"],
                proba if proba is not None else -1.0,
                key,
            )
        else:
            queue_id = 0
            self.logger.info(
                "[SHAPE-BENIGN] high-priority queue=0: %s:%s -> %s:%s "
                "(pkt_count=%d, proba=%.3f)",
                src_ip,
                src_port,
                dst_ip,
                dst_port,
                stats["packet_count"],
                proba if proba is not None else -1.0,
            )

        # Bu noktada kalıcı DROP YOK; sadece kuyruğu belirleyip flow yazıyoruz.
        match = parser.OFPMatch(
            eth_type=0x0800,
            ipv4_src=src_ip,
            ipv4_dst=dst_ip,
            ip_proto=proto,
        )
        self.add_flow_with_queue(
            datapath=datapath,
            priority=50,
            match=match,
            out_port=out_port,
            queue_id=queue_id,
            idle_timeout=60,
            hard_timeout=0,
        )

        # İlk paketi de seçilen queue ile gönderelim
        actions = [
            parser.OFPActionSetQueue(queue_id),
            parser.OFPActionOutput(out_port),
        ]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None,
        )
        datapath.send_msg(out)
