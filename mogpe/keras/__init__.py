#!/usr/bin/env python3
from .experts import SVGPExpert
from .gating_networks import SVGPGatingNetwork

EXPERTS = [SVGPExpert]
GATING_NETWORKS = [SVGPGatingNetwork]

EXPERT_OBJECTS = {expert.__name__: expert for expert in EXPERTS}
GATING_NETWORK_OBJECTS = {
    gating_network.__name__: gating_network for gating_network in GATING_NETWORKS
}
