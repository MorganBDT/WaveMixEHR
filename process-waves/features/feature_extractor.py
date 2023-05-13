"""
feature_extractor.py
--------------------
This module provides a class and methods for pre-processing ECG signals and generating feature vectors using feature
extraction libraries.
--------------------
Adapted from Sebastian D. Goodfellow
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import time
import pandas as pd
import scipy.io as sio
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal

# Local imports
# from features.rri_features import *
# from features.template_features import *
from features.full_waveform_features import *


class Features:

    def __init__(self, fs):
        self.fs = fs

    def extract_features(self, signal_raw, filter_bandwidth, normalize=True, 
                         polarity_check=True, template_before=0.2, template_after=0.4):

        # Preprocess signal
        ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates = self._preprocess_signal(
            signal_raw=signal_raw, filter_bandwidth=filter_bandwidth, normalize=normalize,
            polarity_check=polarity_check, template_before=template_before, template_after=template_after
        )

        # Extract features from waveform
        features = self._group_features(ts=ts, signal_raw=signal_raw,
                                        signal_filtered=signal_filtered, rpeaks=rpeaks,
                                        templates_ts=templates_ts, templates=templates,
                                        template_before=template_before, template_after=template_after)

        return features 

    def _preprocess_signal(self, signal_raw, filter_bandwidth, normalize, polarity_check,
                           template_before, template_after):

        # Filter signal
        signal_filtered = self._apply_filter(signal_raw, filter_bandwidth)

        # Get BioSPPy ECG object
        ecg_object = ecg.ecg(signal=signal_raw, sampling_rate=self.fs, show=False)

        # Get BioSPPy output
        ts = ecg_object['ts']          # Signal time array
        rpeaks = ecg_object['rpeaks']  # rpeak indices

        # Get templates and template time array
        templates, rpeaks = self._extract_templates(signal_filtered, rpeaks, template_before, template_after)
        templates_ts = np.linspace(-template_before, template_after, templates.shape[1], endpoint=False)

        # Polarity check
        signal_raw, signal_filtered, templates = self._check_waveform_polarity(polarity_check=polarity_check,
                                                                               signal_raw=signal_raw,
                                                                               signal_filtered=signal_filtered,
                                                                               templates=templates)
        # Normalize waveform
        signal_raw, signal_filtered, templates = self._normalize_waveform_amplitude(normalize=normalize,
                                                                                    signal_raw=signal_raw,
                                                                                    signal_filtered=signal_filtered,
                                                                                    templates=templates)
        return ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates

    @staticmethod
    def _check_waveform_polarity(polarity_check, signal_raw, signal_filtered, templates):

        """Invert waveform polarity if necessary."""
        if polarity_check:

            # Get extremes of median templates
            templates_min = np.min(np.median(templates, axis=1))
            templates_max = np.max(np.median(templates, axis=1))

            if np.abs(templates_min) > np.abs(templates_max):
                return signal_raw * -1, signal_filtered * -1, templates * -1
            else:
                return signal_raw, signal_filtered, templates

    @staticmethod
    def _normalize_waveform_amplitude(normalize, signal_raw, signal_filtered, templates):
        """Normalize waveform amplitude by the median R-peak amplitude."""
        if normalize:

            # Get median templates max
            templates_max = np.max(np.median(templates, axis=1))

            return signal_raw / templates_max, signal_filtered / templates_max, templates / templates_max

    def _extract_templates(self, signal_filtered, rpeaks, before, after):

        # convert delimiters to samples
        before = int(before * self.fs)
        after = int(after * self.fs)

        # Sort R-Peaks in ascending order
        rpeaks = np.sort(rpeaks)

        # Get number of sample points in waveform
        length = len(signal_filtered)

        # Create empty list for templates
        templates = []

        # Create empty list for new rpeaks that match templates dimension
        rpeaks_new = np.empty(0, dtype=int)

        # Loop through R-Peaks
        for rpeak in rpeaks:

            # Before R-Peak
            a = rpeak - before
            if a < 0:
                continue

            # After R-Peak
            b = rpeak + after
            if b > length:
                break

            # Append template list
            templates.append(signal_filtered[a:b])

            # Append new rpeaks list
            rpeaks_new = np.append(rpeaks_new, rpeak)

        # Convert list to numpy array
        templates = np.array(templates).T

        return templates, rpeaks_new

    def _apply_filter(self, signal_raw, filter_bandwidth):
        """Apply FIR bandpass filter to waveform."""
        signal_filtered, _, _ = filter_signal(signal=signal_raw, ftype='FIR', band='bandpass',
                                              order=int(0.3 * self.fs), frequency=filter_bandwidth,
                                              sampling_rate=self.fs)
        return signal_filtered

    def _group_features(self, ts, signal_raw, signal_filtered, rpeaks,
                        templates_ts, templates, template_before, template_after):

        """Get a dictionary of all ECG features"""

        # Empty features dictionary
        features = dict()

        # Extract features
        full_waveform_features = FullWaveformFeatures(ts=ts, signal_raw=signal_raw,
                                                        signal_filtered=signal_filtered, rpeaks=rpeaks,
                                                        templates_ts=templates_ts, templates=templates,
                                                        fs=self.fs)
        full_waveform_features.extract_full_waveform_features()

        # Update feature dictionary
        features.update(full_waveform_features.get_full_waveform_features())

        return features
