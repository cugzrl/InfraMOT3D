"""按顺序运行本轮离线实验"""

import audit_data
import build_knowledge
import eval_association
import eval_prediction
import test_core
import write_summary


def main():
    test_core.test_local_roundtrip()
    test_core.test_interpolate()
    test_core.test_cv_straight()
    test_core.test_forbidden_overlap()
    test_core.test_config_frozen()
    audit_data.main()
    build_knowledge.main()
    eval_prediction.main()
    eval_association.main()
    write_summary.main()


if __name__ == "__main__":
    main()
