from .student_t import StudentTEnergy
from .anti_collapse import AntiCollapseRegularizer
from .masking import complementary_element_mask, make_window_mask
from .windowing import windowize
from .rea import ResidualEvidenceAggregator

# Legacy public name retained for downstream compatibility.
MRResa = ResidualEvidenceAggregator
__all__ = ['StudentTEnergy','AntiCollapseRegularizer',
           'complementary_element_mask','make_window_mask','windowize',
           'ResidualEvidenceAggregator','MRResa']
