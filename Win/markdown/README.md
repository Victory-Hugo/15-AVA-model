# AVA Model: Autochthonous vs Admixed Population Analysis

## English Version

### Overview
The AVA (Autochthonous vs Admixed) Model is a comprehensive analytical pipeline designed to evaluate and classify populations based on their genetic diversity patterns, temporal distribution characteristics, and haplogroup uniqueness. The system integrates multiple statistical approaches to distinguish between autochthonous (indigenous/origin-like), admixed (mixed), and intermediate population types.

### Pipeline Architecture
The analysis pipeline consists of four main components executed sequentially:

1. **TMRCA Statistical Analysis** (`1-divergence_time_shape.py`)
2. **AMOVA Diversity Pattern Analysis** (`2-AMOVA.py`)
3. **Haplogroup Uniqueness Analysis** (`3-unique_haplogroup.py`)
4. **Integrated Scoring and Classification** (`4-score.py`)

---

### Input Files Requirements

#### 1. Primary Dataset (`Example/1-ID_Time_Class.csv`)
**Required columns:**
- `Continent`: Geographic classification of samples
- `Time_years`: TMRCA values in years
- Additional demographic columns as needed

**Format:** CSV with headers
**Purpose:** Main dataset for TMRCA distribution analysis

**Example structure:**
```csv
ID,Time_years,Continent
Sample1,51666.77778,Africa
Sample2,11777.77778,Africa
...
```

#### 2. AMOVA Results (`Example/2-AMOVA.csv`)
**Required columns:**
- `Continent`: Geographic classification
- `Source of variation`: Type of variation (must include "Within populations")
- `Percentage of variation`: Numerical percentage values

**Format:** CSV/TSV (auto-detected separator)
**Purpose:** Population genetic diversity structure analysis

**Example structure:**
```csv
Continent,Source of variation,Percentage of variation
Africa,Among groups,15.2
Africa,Within populations,84.8
Europe,Among groups,12.7
Europe,Within populations,87.3
...
```

#### 3. Public Haplogroup Dataset (`Example/3-public.csv`)
**Required columns:**
- `ID`: Sample identifier
- `Continent`: Geographic classification
- `Haplogroup`: Mitochondrial haplogroup designation

**Format:** CSV with headers
**Purpose:** Haplogroup frequency and uniqueness analysis

**Example structure:**
```csv
ID,Continent,Haplogroup
Sample1,Africa,L3f1b2
Sample2,Africa,L3f1b2
Sample3,Africa,L3f1b2
Sample4,Africa,L3f1b2
Sample5,Africa,L3f1b4a1
...
```

---

### Output Files Description

#### Primary Output Directory: `output/`

#### 1. TMRCA Statistics (`Final_tmrca_stats.csv`)
**Content:** Statistical summary of TMRCA distributions by continent
**Columns:**
- `Continent`: Geographic group
- `Count`: Sample size
- `Max`, `Min`, `Range`: Distribution bounds
- `StdDev`: Standard deviation
- `Skewness`: Distribution asymmetry
- `AncientRatio(>100000)`: Proportion of ancient lineages
- `Estimated_Peaks`: Number of estimated distribution modes

#### 2. AMOVA Diversity Scores (`Final_AMOVA_scores.csv`)
**Content:** Within-population diversity patterns
**Columns:**
- `Continent`: Geographic group
- `Diversity_pattern_score`: Normalized diversity score (0-1)

#### 3. Haplogroup Uniqueness Scores (`Final_unique_hap.csv`)
**Content:** Haplogroup uniqueness metrics by region
**Columns:**
- `Continent`: Geographic group
- `Unique_hap_score`: Uniqueness score (0-1)

#### 4. Final Integrated Scores (`Final_metrics_scored.csv`)
**Content:** Comprehensive scoring and classification results
**Key Columns:**
- `Continent`: Geographic group
- `All_score`: Final integrated score (0-1)
- `Class_label`: Classification result
  - `Origin_like`: Score > 0.7 (autochthonous characteristics)
  - `Middle_like`: 0.3 < Score ≤ 0.7 (intermediate characteristics)
  - `Mix_like`: Score ≤ 0.3 (admixed characteristics)

#### Secondary Output Directory: `output/Frequency_result/`

#### 5. Detailed Haplogroup Analysis Files
- `unique_haplogroups_[Region].txt`: List of unique haplogroups per region
- `merged_haplogroup_frequencies_[Region].txt`: Frequency tables with population counts
- `score_[Region].csv`: Individual haplogroup scoring results
- `combined_haplogroup_scores.csv`: Comprehensive cross-regional comparison

---

### Usage Instructions

#### Basic Execution
```bash
cd /path/to/15-AVA-model/script/
bash 1-pipe.sh
```

#### Custom Parameters
The pipeline can be customized by modifying parameters in `1-pipe.sh`:

**TMRCA Analysis Parameters:**
- `--ancient_threshold`: Threshold for ancient lineage classification (default: 100000)
- `--gmm_max_components`: Maximum Gaussian mixture components (default: 5)
- `--skew_method`: Skewness calculation method (auto/moment/quantile)

**Scoring Weights:**
- `--w-max`, `--w-ancient`: TMRCA-related weights (default: 2.0)
- `--w-diversity`: AMOVA diversity weight (default: 1.5)
- `--w-unique`: Haplogroup uniqueness weight (default: 1.5)

**Classification Thresholds:**
- `--thr-origin`: Origin-like threshold (default: 0.7)
- `--thr-mix-low`: Mixed-like threshold (default: 0.3)

---

### Technical Requirements

#### Dependencies
- Python 3.7+
- pandas
- numpy
- scipy
- scikit-learn
- argparse
- pathlib

#### System Requirements
- Memory: 4GB+ recommended for large datasets
- Storage: Sufficient space for output files
- OS: Linux/macOS/Windows with bash support

---

### Interpretation Guidelines

#### Score Interpretation
- **High scores (>0.7):** Suggest autochthonous/origin-like characteristics
  - High temporal diversity
  - Low within-population genetic diversity
  - Unique haplogroup signatures
  
- **Medium scores (0.3-0.7):** Indicate intermediate characteristics
  - Balanced temporal and genetic diversity
  - Mixed haplogroup patterns

- **Low scores (<0.3):** Suggest admixed characteristics
  - Limited temporal diversity
  - High within-population genetic diversity
  - Common/shared haplogroup patterns

#### Statistical Considerations
- Results should be interpreted in conjunction with archaeological and historical evidence
- Sample size effects should be considered when comparing populations
- Geographic and temporal sampling biases may affect results


