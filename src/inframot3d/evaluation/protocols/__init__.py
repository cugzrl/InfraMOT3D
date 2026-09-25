from inframot3d.evaluation.protocols.v2xseq import V2XSeqProtocol

PROTOCOLS = {"v2xseq": V2XSeqProtocol}


def build_protocol(name, root):
    if name not in PROTOCOLS:
        raise KeyError("缺少评估协议%s" % name)
    return PROTOCOLS[name](root)
