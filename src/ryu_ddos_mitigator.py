# src/ryu_ddos_mitigator.py
# (Sürekli Saldırı Tespiti İçin "Auto-Reset" Özelliği Eklendi)

import sys
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, icmp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, META_PATH

class DDoSMitigator(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DDoSMitigator, self).__init__(*args, **kwargs)

        # ----------------- ML modeli yükle -----------------
        model_path = Path(MODEL_PATH)
        meta_path = Path(META_PATH)

        if not model_path.exists():
            self.logger.error("ML model not found: %s", model_path)
            sys.exit(1)
            
        self.logger.info("Loading model from %s...", model_path)
        self.model = joblib.load(model_path)
        
        self.feature_order = []
        self.threshold = 0.5 

        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            self.feature_order = meta.get("feature_order", [])
            meta_threshold = float(meta.get("threshold", 0.5))
            if meta_threshold < 0.1:
                self.threshold = 0.5
            else:
                self.threshold = meta_threshold
        else:
            self.logger.warning("Meta file not found. Using defaults.")

        # Flow tracking (IP bazlı)
        self.flow_stats = {}
        self.mac_to_port = {}
        
        self.min_pkts_for_ml = 3 
        
        # İstatistikleri kaç pakette bir sıfırlayalım?
        # Bu sayı saldırı paterninin bozulmadan yakalanabileceği bir aralık olmalı.
        self.reset_interval = 100

        self.logger.info("="*60)
        self.logger.info("CIC-IDS2019 DDoS Mitigator (Auto-Reset Enabled)")
        self.logger.info("Threshold: %.2f", self.threshold)
        self.logger.info("Reset Interval: Every %d packets", self.reset_interval)
        self.logger.info("="*60)

    def add_flow_with_queue(self, datapath, priority, match, out_port, 
                           queue_id, idle_timeout=10, hard_timeout=60):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser

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

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        self.logger.info("Switch connected: dpid=%s", datapath.id)

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=0, match=match, instructions=inst)
        datapath.send_msg(mod)

    def _update_flow_stats(self, dpid, src, dst, proto, pkt_len, tcp_flags=None):
        key = (dpid, src, dst) 
        now = time.time()

        st = self.flow_stats.get(key)
        if st is None:
            st = {
                "first_seen": now,
                "last_seen": now,
                "fwd_packet_count": 0,
                "bwd_packet_count": 0,
                "fwd_bytes": 0,
                "bwd_bytes": 0,
                "syn_count": 0,
                "rst_count": 0,
                "ack_count": 0,
            }
            self.flow_stats[key] = st

        st["last_seen"] = now
        st["fwd_packet_count"] += 1
        st["fwd_bytes"] += pkt_len

        if tcp_flags is not None:
            if tcp_flags & 0x02: st["syn_count"] += 1
            if tcp_flags & 0x04: st["rst_count"] += 1
            if tcp_flags & 0x10: st["ack_count"] += 1

        return key, st

    def _build_feature_vector_optimized(self, stats):
        duration = max(0.001, stats["last_seen"] - stats["first_seen"])
        
        fwd_packets = float(stats["fwd_packet_count"])
        bwd_packets = float(stats.get("bwd_packet_count", 0))
        fwd_bytes = float(stats["fwd_bytes"])
        bwd_bytes = float(stats.get("bwd_bytes", 0))
        
        flow_packets_per_sec = (fwd_packets + bwd_packets) / duration
        flow_bytes_per_sec = (fwd_bytes + bwd_bytes) / duration
        
        syn_count = float(stats["syn_count"])
        rst_count = float(stats["rst_count"])
        ack_count = float(stats["ack_count"])

        feat_map = {
            "Flow Duration": duration * 1000000,
            "Total Length of Fwd Packets": fwd_bytes,
            "Total Length of Bwd Packets": bwd_bytes,
            "Total Fwd Packets": fwd_packets,
            "Total Backward Packets": bwd_packets,
            "Flow Packets/s": flow_packets_per_sec,
            "Flow Bytes/s": flow_bytes_per_sec,
            "SYN Flag Count": syn_count,
            "RST Flag Count": rst_count,
            "ACK Flag Count": ack_count,
        }

        feat_map["bytes_ratio"] = (fwd_bytes + 1.0) / (bwd_bytes + 1.0)
        feat_map["packet_ratio"] = (fwd_packets + 1.0) / (bwd_packets + 1.0)
        feat_map["log_flow_duration"] = np.log1p(feat_map["Flow Duration"])
        feat_map["log_fwd_bytes"] = np.log1p(fwd_bytes)
        feat_map["log_bwd_bytes"] = np.log1p(bwd_bytes)
        
        total_flags = syn_count + rst_count + ack_count + 1.0
        feat_map["syn_ratio"] = syn_count / total_flags
        feat_map["rst_ratio"] = rst_count / total_flags
        feat_map["ack_ratio"] = ack_count / total_flags
        feat_map["estimated_packets"] = flow_packets_per_sec * duration

        vector = []
        for name in self.feature_order:
            val = feat_map.get(name, 0.0)
            if np.isinf(val) or np.isnan(val): val = 0.0
            vector.append(val)
            
        return np.array([vector], dtype=np.float32)

    def _predict_attack(self, X):
        try:
            if hasattr(self.model, "predict_proba"):
                proba = float(self.model.predict_proba(X)[0, 1])
                is_attack = (proba >= self.threshold)
                return (1 if is_attack else 0), proba
            else:
                y = int(self.model.predict(X)[0])
                return y, None
        except Exception as e:
            self.logger.error("Prediction error: %s", e)
            return 0, 0.0

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

        if eth.ethertype in [0x88cc, 0x86dd]:
            return

        src_mac = eth.src
        dst_mac = eth.dst

        # L2 Learning
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port

        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            out_port = ofp.OFPP_FLOOD

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt is None:
            actions = [parser.OFPActionOutput(out_port)]
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port, actions=actions, data=msg.data)
            datapath.send_msg(out)
            return

        src_ip = ip_pkt.src
        dst_ip = ip_pkt.dst
        proto = ip_pkt.proto
        tcp_flags = None

        t = pkt.get_protocol(tcp.tcp)
        u = pkt.get_protocol(udp.udp)
        if t: tcp_flags = t.bits

        # Stats Güncelleme
        pkt_len = len(msg.data)
        key, stats = self._update_flow_stats(dpid, src_ip, dst_ip, proto, pkt_len, tcp_flags)

        # Warm-up
        if stats["fwd_packet_count"] < self.min_pkts_for_ml:
            actions = [parser.OFPActionOutput(out_port)]
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port, actions=actions, data=msg.data)
            datapath.send_msg(out)
            return

        # Tahmin
        X = self._build_feature_vector_optimized(stats)
        y_pred, proba = self._predict_attack(X)

        if y_pred == 1:
            queue_id = 1
            priority = 100
            # Sadece riskli durumlarda log bas (Terminal kirliliğini önlemek için)
            if proba > self.threshold:
                self.logger.warning("!!! [ATTACK] %s -> %s | Pkts: %d | Prob: %.2f", 
                                    src_ip, dst_ip, stats["fwd_packet_count"], proba)
        else:
            queue_id = 0
            priority = 50
            if stats["fwd_packet_count"] % 20 == 0: # Sadece 20 pakette bir normal logu bas
                self.logger.info("[BENIGN] %s -> %s | Pkts: %d | Prob: %.2f", 
                                 src_ip, dst_ip, stats["fwd_packet_count"], proba)

        # --- YENİ EKLENEN KISIM: İSTATİSTİK SIFIRLAMA ---
        # Eğer paket sayısı reset limitini geçtiyse, istatistikleri temizle.
        # Bu sayede bir sonraki paket "yeni bir akışın başlangıcı" gibi işlem görür 
        # ve model tekrar "ATTACK" kararı verebilir.
        if stats["fwd_packet_count"] >= self.reset_interval:
            # self.logger.debug("Resetting stats for flow %s -> %s", src_ip, dst_ip)
            del self.flow_stats[key]
        # -----------------------------------------------

        # Kural Ekleme
        match = parser.OFPMatch(
            eth_type=0x0800,
            ipv4_src=src_ip,
            ipv4_dst=dst_ip,
            ip_proto=proto
        )
        
        self.add_flow_with_queue(
            datapath=datapath,
            priority=priority,
            match=match,
            out_port=out_port,
            queue_id=queue_id,
            idle_timeout=10, 
            hard_timeout=60
        )

        actions = [parser.OFPActionSetQueue(queue_id), parser.OFPActionOutput(out_port)]
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port, actions=actions, data=msg.data)
        datapath.send_msg(out)