"""Path A reuses the frozen kernel estimator and only changes construction."""
import time
from sprint_evaluator import SprintEvaluator
from upgrade_partition import split_scores


class EquivalentPartitionEvaluator(SprintEvaluator):
    def prepare(self, q, budget, cfg, control=None):
        if (cfg['score'] != 'projection64' or cfg['grouping'] != 'quantile'
                or not cfg['full_target'] or cfg['allocation'] != 'proportional'
                or control is not None):
            raise ValueError('Path A supports only the frozen final estimator')
        if self.projection is None:
            raise RuntimeError('Build projection offline')
        start = time.perf_counter_ns()
        z = self.projection.query(q)
        k = min(int(budget*cfg['H_fraction']), self.n, max(0, budget-1))
        H, cells, detail = split_scores(z['d2hat'], self.id_array, k, cfg['J'])
        stages = dict(projection_ns=z['projection_ns'],
                      low_dim_score_ns=z['low_dim_score_ns'],
                      sorting_and_strata_ns=detail['partition_total_ns'],
                      residual_build_ns=0)
        # Breakdown values go in metadata, not overlapping additive phases.
        counts = dict(full_dimension_encoder_inner_products=0,
            full_dimension_projection_inner_products=64,
            low_dim_score_evaluations=self.n, low_dim_score_dimension=64,
            hash_bucket_visits=0, retrieved_posting_ids=0, code_trie_node_visits=0,
            norm_posting_ids=0, hamming_table_comparisons=0,
            full_dimension_query_norm_evaluations=0, residual_id_scan_count=0,
            **detail)
        return dict(H=H, cells=cells, C=self.id_array, residual_size=0,
                    stages=stages, counts=counts,
                    prepare_ns=time.perf_counter_ns()-start)
