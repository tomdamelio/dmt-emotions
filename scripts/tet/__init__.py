# -*- coding: utf-8 -*-
"""TET (Temporal Experience Tracking) Analysis Module

This module provides functionality for loading, validating, and analyzing
Temporal Experience Tracking (TET) data from DMT psychopharmacology studies.

The module includes:
    - TETDataLoader: Load TET data from CSV or .mat files
    - TETDataValidator: Validate data integrity and quality
    - ValidationReporter: Generate validation reports

Data Format:
    TET data consists of 15 phenomenological dimensions rated continuously
    during psychedelic experiences. The .mat files contain a 'dimensions'
    matrix of shape (n_bins, 15) where columns represent the 15 dimensions
    in a specific order. See config.TET_DIMENSION_COLUMNS for the complete
    list and descriptions, or docs/PIPELINE.md for detailed documentation.

Example:
    >>> from tet.data_loader import TETDataLoader
    >>> from tet.validator import TETDataValidator
    >>> from tet.reporter import ValidationReporter
    >>> 
    >>> # Load data
    >>> loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
    >>> data = loader.load_data()
    >>> 
    >>> # Validate
    >>> validator = TETDataValidator(data, dimension_columns)
    >>> results = validator.validate_all()
    >>> 
    >>> # Generate report
    >>> reporter = ValidationReporter(results, 'results/tet/validation')
    >>> report_path = reporter.generate_report()
"""

__version__ = "0.1.0"
