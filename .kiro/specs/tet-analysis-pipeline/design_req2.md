# Design Document: TET Preprocessing and Standardization (Requirement 2)

## Overview

This design document describes the technical architecture for implementing Requirement 2: Data Preprocessing and Standardization of the TET Analysis Pipeline. This component is responsible for trimming sessions to analysis windows, computing within-subject standardization, and creating composite indices.

## Architecture

The preprocessing system follows a functional pipeline architecture with three main stages:

1. **Session Trimming**: Trim sessions to analysis time windows (0-10 min RS, 0-20 min DMT)
2. **Standardization**: Compute global within-subject z-scores across all dimensions and sessions
3. **Composite Index Creation**: Derive affect, imagery, and self integration indices

## Components and Interfaces

### 1. TETPreprocessor

**Purpose**: Preprocess and standardize TET data within subjects

**Interface**:
```python
class TETPreprocessor:
    def __init__(self, data: pd.DataFrame, dimension_columns: List[str]):
        """Initialize preprocessor with validated TET data"""
        
    def preprocess_all(self) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline
        
        Returns:
            DataFrame with trimmed sessions, z-scored dimensions, and composite indices
        """
        
    def trim_sessions(self) -> pd.DataFrame:
        """Trim RS to 0-10 min (bins 0-19), DMT to 0-20 min (bins 0-39)"""
        
    def create_valence_variables(self) -> pd.DataFrame:
        """Create valence_pos (pleasantness) and valence_neg (unpleasantness)"""
        
    def standardize_within_subject(self) -> pd.DataFrame:
        """
        Compute global within-subject z-scores
        
        For each subject, compute z-scores using global mean and std
        across all 15 dimensions and all 4 sessions
        """
        
    def create_composite_indices(self) -> pd.DataFrame:
        """
        Create affect_index_z, imagery_index_z, and self_index_z
        
        - affect_index_z = mean(z_pleasantness, z_bliss) - mean(z_anxiety, z_unpleasantness)
        - imagery_index_z = mean(z_elementary_imagery, z_complex_imagery)
        - self_index_z = -z_disembodiment + z_selfhood (higher = more integrated)
        """
```

**Implementation Details**:

**trim_sessions()**:
- Filter data where:
  - `(state == 'RS') & (t_sec < 600)` OR  # 0-10 minutes
  - `(state == 'DMT') & (t_sec < 1200)`   # 0-20 minutes
- Preserves original 0.25 Hz sampling rate (1 point every 4 seconds)
- RS: 150 points, DMT: 300 points
- Preserves all columns
- Returns trimmed DataFrame

**create_valence_variables()**:
- Creates `valence_pos` column = `pleasantness`
- Creates `valence_neg` column = `unpleasantness`
- These are aliases for easier interpretation

**standardize_within_subject()**:
- For each subject:
  1. Stack all 15 dimension values across all 4 sessions into single array
  2. Compute global_mean = mean of all values
  3. Compute global_std = std of all values
  4. For each dimension: `z_dimension = (dimension - global_mean) / global_std`
- Creates new columns with `_z` suffix (e.g., `pleasantness_z`)
- Preserves original raw columns for descriptive statistics
- This approach controls for individual differences in scale usage patterns

**create_composite_indices()**:
- **affect_index_z**:
  ```python
  positive = (pleasantness_z + bliss_z) / 2
  negative = (anxiety_z + unpleasantness_z) / 2
  affect_index_z = positive - negative
  ```
  
- **imagery_index_z**:
  ```python
  imagery_index_z = (elementary_imagery_z + complex_imagery_z) / 2
  ```
  
- **self_index_z**:
  ```python
  self_index_z = -disembodiment_z + selfhood_z
  # Higher values = more self-integration (less disembodiment, more selfhood)
  ```

### 2. PreprocessingMetadata

**Purpose**: Document preprocessing parameters and composite index definitions

**Interface**:
```python
class PreprocessingMetadata:
    def __init__(self):
        """Initialize metadata container"""
        
    def generate_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate preprocessing metadata
        
        Returns:
            Dictionary with preprocessing parameters and composite definitions
        """
```

**Implementation Details**:
- Documents:
  - Trimming windows (RS: 0-10 min, DMT: 0-20 min)
  - Standardization method (global within-subject)
  - Composite index formulas with directionality
  - Number of subjects, sessions, time bins after preprocessing
  - Timestamp of preprocessing

## Data Models

### Input Data Schema

Expected input from TETDataLoader (after validation):
```
subject,session_id,state,dose,t_bin,t_sec,pleasantness,unpleasantness,...
S01,1,RS,Low,0,0,5.2,1.3,...
S01,1,RS,Low,1,4,5.5,1.1,...
```

### Output Data Schema

After preprocessing:
```
subject,session_id,state,dose,t_bin,t_sec,
pleasantness,pleasantness_z,
unpleasantness,unpleasantness_z,
...(all 15 dimensions with raw and _z versions),
valence_pos,valence_neg,
affect_index_z,imagery_index_z,self_index_z
```

**New Columns**:
- `{dimension}_z`: Z-scored version of each dimension (15 columns)
- `valence_pos`: Alias for pleasantness
- `valence_neg`: Alias for unpleasantness
- `affect_index_z`: Composite affect index
- `imagery_index_z`: Composite imagery index
- `self_index_z`: Composite self-integration index

## Standardization Algorithm

### Global Within-Subject Z-Score

**Rationale**: Controls for individual differences in how subjects use the rating scale (e.g., some subjects may use the full 0-10 range while others cluster around 5).

**Algorithm**:
```python
def standardize_within_subject(data, subject_id, dimension_columns):
    # Get all data for this subject (all sessions, all dimensions)
    subject_data = data[data['subject'] == subject_id]
    
    # Stack all dimension values into single array
    all_values = subject_data[dimension_columns].values.flatten()
    
    # Compute global statistics
    global_mean = np.mean(all_values)
    global_std = np.std(all_values, ddof=1)
    
    # Standardize each dimension
    for dim in dimension_columns:
        z_col = f"{dim}_z"
        data.loc[data['subject'] == subject_id, z_col] = (
            (subject_data[dim] - global_mean) / global_std
        )
    
    return data
```

**Properties**:
- Mean of all z-scored values for a subject ≈ 0
- Std of all z-scored values for a subject ≈ 1
- Preserves relative differences between dimensions within subject
- Removes subject-specific scale usage bias

## Composite Index Formulas

### affect_index_z

**Formula**: `mean(pleasantness_z, bliss_z) - mean(anxiety_z, unpleasantness_z)`

**Interpretation**:
- Positive values: More positive affect (pleasant, blissful)
- Negative values: More negative affect (anxious, unpleasant)
- Range: Approximately -4 to +4 (in z-score units)

**Components**:
- Positive pole: pleasantness, bliss
- Negative pole: anxiety, unpleasantness
- Note: emotional_intensity is NOT included (can be high for both positive and negative states)

### imagery_index_z

**Formula**: `mean(elementary_imagery_z, complex_imagery_z)`

**Interpretation**:
- Higher values: More vivid imagery (both elementary and complex)
- Lower values: Less imagery
- Range: Approximately -3 to +3 (in z-score units)

### self_index_z

**Formula**: `-disembodiment_z + selfhood_z`

**Interpretation**:
- Higher values: More self-integration (embodied, strong sense of self)
- Lower values: Less self-integration (disembodied, ego dissolution)
- Range: Approximately -4 to +4 (in z-score units)

**Directionality**:
- disembodiment is inverted (multiplied by -1) so higher values = more embodied
- selfhood is positive, so higher values = stronger sense of self

## Error Handling

### Missing Data
- If subject has < 4 sessions: Log warning, standardize with available sessions
- If dimension has all NaN for a subject: Set z-score to NaN, log warning

### Edge Cases
- If global_std = 0 for a subject (all values identical): Set all z-scores to 0, log warning
- If session has < expected bins after trimming: Keep available bins, log info

## Testing Strategy

### Unit Tests

**test_preprocessor.py**:
- Test session trimming with correct bin ranges
- Test global within-subject standardization
  - Verify mean ≈ 0 and std ≈ 1 across all dimensions for each subject
  - Verify z-scores preserve relative differences
- Test composite index calculations
  - Verify affect_index_z formula
  - Verify imagery_index_z formula
  - Verify self_index_z formula and directionality
- Test valence variable creation

**test_preprocessing_metadata.py**:
- Test metadata generation
- Verify all required fields present
- Test JSON serialization

### Integration Tests

**test_preprocessing_pipeline.py**:
- Test complete preprocessing pipeline
- Verify output DataFrame structure
- Verify all new columns created
- Test with realistic TET data

## Dependencies

**Required packages**:
- `pandas >= 1.5.0`: Data manipulation
- `numpy >= 1.24.0`: Numerical operations
- `typing`: Type hints

**Project modules**:
- `config.py`: TET configuration
- `tet.data_loader`: Data loading
- `tet.validator`: Data validation

## File Structure

```
dmt-emotions/
├── scripts/
│   ├── tet/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── validator.py
│   │   ├── preprocessor.py          # NEW: TETPreprocessor class
│   │   └── metadata.py              # NEW: PreprocessingMetadata class
│   └── preprocess_tet_data.py       # NEW: Main preprocessing script
└── results/
    └── tet/
        ├── preprocessed/
        │   ├── tet_preprocessed.csv
        │   └── preprocessing_metadata.json
        └── validation/
```

## Usage Example

```python
from tet.data_loader import TETDataLoader
from tet.validator import TETDataValidator
from tet.preprocessor import TETPreprocessor
from tet.metadata import PreprocessingMetadata
import config

# Load and validate data
loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
data = loader.load_data()

validator = TETDataValidator(data, config.TET_DIMENSION_COLUMNS)
validation_results = validator.validate_all()

# Apply clamping if needed
if len(validation_results['range_violations']) > 0:
    data, _ = validator.clamp_out_of_range_values()

# Preprocess
preprocessor = TETPreprocessor(data, config.TET_DIMENSION_COLUMNS)
data_preprocessed = preprocessor.preprocess_all()

# Generate metadata
metadata_gen = PreprocessingMetadata()
metadata = metadata_gen.generate_metadata(data_preprocessed)

# Save outputs
data_preprocessed.to_csv('results/tet/preprocessed/tet_preprocessed.csv', index=False)

import json
with open('results/tet/preprocessed/preprocessing_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Preprocessing complete. Output shape: {data_preprocessed.shape}")
print(f"New columns: {[col for col in data_preprocessed.columns if '_z' in col or 'index' in col]}")
```

## Performance Considerations

- Expected data size after trimming (@ 0.25 Hz):
  - RS: 18 subjects × 2 sessions × 150 points = 5,400 rows
  - DMT: 18 subjects × 2 sessions × 300 points = 10,800 rows
  - Total: ~16,200 rows × 35 columns ≈ 567,000 cells
- Standardization is vectorized using pandas/numpy for efficiency
- Memory usage: ~5-10 MB for full dataset
- Processing time: < 3 seconds on typical hardware
