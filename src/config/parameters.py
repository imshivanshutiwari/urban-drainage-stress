"""Default model and data parameters.

All defaults are conservative and should be reviewed event-by-event.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class RainfallConfig:
    gauge_min_coverage: float = 0.75  # fraction of gauges reporting to trust
    radar_bias_correction: bool = True
    temporal_resolution_minutes: int = 15
    accumulation_window_hours: int = 6
    intensity_threshold_mm_hr: float = 10.0  # highlight intense cells


@dataclass
class ComplaintConfig:
    min_fields_required: List[str] = (
        "timestamp",
        "latitude",
        "longitude",
        "complaint_type",
    )
    deduplicate_minutes: int = 60  # collapse near-duplicates within window
    geo_epsilon_meters: float = 75.0  # spatial tolerance for duplicates
    spatial_uncertainty_radius_m: float = 300.0  # radius for coarse wards
    delay_prior_mean_hours: float = 2.0  # prior mean reporting delay
    delay_prior_sd_hours: float = 1.5  # prior sd for delay (lognormal)
    max_delay_hours: float = 48.0  # cap delays to avoid unrealistic windows
    time_aggregation_minutes: int = 60  # default aggregation window


@dataclass
class TerrainConfig:
    dem_resolution_meters: float = 10.0
    fill_sinks: bool = True
    slope_threshold_degrees: float = 0.5  # low-slope areas prone to ponding
    sink_fill_max_depth_m: float = 0.5  # cap sink filling to avoid smoothing
    slope_smooth_kernel: int = 3  # bounded kernel for slope smoothing
    spike_zscore_threshold: float = 6.0  # detect extreme spikes
    flat_slope_threshold: float = 0.1  # deg; near-zero slopes flagged as flats
    accumulation_confidence_floor: float = 0.4  # min confidence in outputs


@dataclass
class LatentStressModelConfig:
    spatial_corr_length_m: float = 1000.0
    temporal_corr_minutes: int = 30
    prior_stress_mean: float = 0.0
    prior_stress_sd: float = 1.0
    observation_noise_sd: float = 0.5
    weight_intensity: float = 1.0
    weight_accumulation: float = 0.4  # Weight for windowed rainfall (not cumulative)
    weight_upstream: float = 0.8
    intercept: float = 0.0
    memory_decay: float = 0.35  # 0=no memory, closer to 1=long persistence
    variance_floor: float = 1e-4
    missing_data_tolerance: float = 0.7  # min fraction of non-nan inputs
    upstream_confidence_floor: float = 0.5
    missing_variance_inflation: float = 2.0


@dataclass
class ObservationModelConfig:
    base_reporting_prob: float = 0.3
    min_reporting_prob: float = 0.01
    max_reporting_prob: float = 0.95
    delay_mean_hours: float = 2.0
    delay_sd_hours: float = 1.5
    delay_max_hours: float = 48.0
    rate_floor: float = 1e-6
    coverage_threshold: float = 0.05


@dataclass
class InferenceConfig:
    cred_interval: float = 0.9
    min_event_duration_minutes: int = 30
    anomaly_zscore: float = 2.0
    posterior_variance_floor: float = 1e-6


@dataclass
class ValidationConfig:
    rain_extreme_quantile: float = 0.9
    lead_time_threshold_minutes: int = 120
    stress_jump_zscore: float = 3.0
    low_lying_quantile: float = 0.2
    overconfident_threshold: float = 0.1
    underconfident_threshold: float = 0.5
    stress_collapse_factor: float = 0.2
    terrain_shuffle_drop: float = 0.2


@dataclass
class OutputsConfig:
    high_risk_threshold: float = 0.8
    medium_risk_threshold: float = 0.5
    uncertainty_high: float = 0.4
    uncertainty_low: float = 0.1
    crs: str = "EPSG:4326"
    driver: str = "GeoJSON"
    html_max_points: int = 5000
    snapshot_times: tuple[str, ...] = ("pre", "peak", "post")


@dataclass
class Parameters:
    rainfall: RainfallConfig = RainfallConfig()
    complaints: ComplaintConfig = ComplaintConfig()
    terrain: TerrainConfig = TerrainConfig()
    latent_model: LatentStressModelConfig = LatentStressModelConfig()
    observation: ObservationModelConfig = ObservationModelConfig()
    inference: InferenceConfig = InferenceConfig()
    validation: ValidationConfig = ValidationConfig()
    outputs: OutputsConfig = OutputsConfig()


def get_default_parameters() -> Parameters:
    """Return a Parameters instance, loading from config/roi_config.yaml if available."""
    import yaml
    from pathlib import Path

    # Default conservative parameters
    params = Parameters()
    
    # Try to load from YAML
    # Path is relative to project root. 
    # Provided we run from root, or relative to this file.
    # config/roi_config.yaml is in .../urban_drainage_stress/config/roi_config.yaml
    # src/config/parameters.py is in .../urban_drainage_stress/src/config/parameters.py
    # So relevant path is ../../config/roi_config.yaml
    
    config_path = Path(__file__).parent.parent.parent / "config" / "roi_config.yaml"
    
    if not config_path.exists():
        # Fallback to defaults (already initialized)
        return params

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        
        if not data:
            return params
            
        # Update RainfallConfig
        if "rainfall" in data:
            r = data["rainfall"]
            params.rainfall = RainfallConfig(
                gauge_min_coverage=r.get("gauge_min_coverage", 0.75),
                radar_bias_correction=r.get("radar_bias_correction", True),
                temporal_resolution_minutes=r.get("temporal_resolution_minutes", 15),
                accumulation_window_hours=r.get("accumulation_window_hours", 6),
                intensity_threshold_mm_hr=r.get("intensity_threshold_mm_hr", 10.0),
            )
            
        # Update ComplaintConfig
        if "complaints" in data:
            c = data["complaints"]
            params.complaints = ComplaintConfig(
                min_fields_required=c.get("min_fields_required", params.complaints.min_fields_required),
                deduplicate_minutes=c.get("deduplicate_minutes", 60),
                geo_epsilon_meters=c.get("geo_epsilon_meters", 75.0),
                spatial_uncertainty_radius_m=c.get("spatial_uncertainty_radius_m", 300.0),
                delay_prior_mean_hours=c.get("delay_prior_mean_hours", 2.0),
                delay_prior_sd_hours=c.get("delay_prior_sd_hours", 1.5),
                max_delay_hours=c.get("max_delay_hours", 48.0),
                time_aggregation_minutes=c.get("time_aggregation_minutes", 60),
            )
            
        # Update TerrainConfig
        if "terrain" in data:
            t = data["terrain"]
            params.terrain = TerrainConfig(
                dem_resolution_meters=t.get("dem_resolution_meters", 10.0),
                fill_sinks=t.get("fill_sinks", True),
                slope_threshold_degrees=t.get("slope_threshold_degrees", 0.5),
                sink_fill_max_depth_m=t.get("sink_fill_max_depth_m", 0.5),
                slope_smooth_kernel=t.get("slope_smooth_kernel", 3),
                spike_zscore_threshold=t.get("spike_zscore_threshold", 6.0),
                flat_slope_threshold=t.get("flat_slope_threshold", 0.1),
                accumulation_confidence_floor=t.get("accumulation_confidence_floor", 0.4),
            )

        # Update LatentStressModelConfig
        if "latent_model" in data:
            lm = data["latent_model"]
            params.latent_model = LatentStressModelConfig(
                spatial_corr_length_m=lm.get("spatial_corr_length_m", 1000.0),
                temporal_corr_minutes=lm.get("temporal_corr_minutes", 30),
                prior_stress_mean=lm.get("prior_stress_mean", 0.0),
                prior_stress_sd=lm.get("prior_stress_sd", 1.0),
                observation_noise_sd=lm.get("observation_noise_sd", 0.5),
                weight_intensity=lm.get("weight_intensity", 1.0),
                weight_accumulation=lm.get("weight_accumulation", 0.4),
                weight_upstream=lm.get("weight_upstream", 0.8),
                intercept=lm.get("intercept", 0.0),
                memory_decay=lm.get("memory_decay", 0.35),
                variance_floor=lm.get("variance_floor", 1e-4),
                missing_data_tolerance=lm.get("missing_data_tolerance", 0.7),
                upstream_confidence_floor=lm.get("upstream_confidence_floor", 0.5),
                missing_variance_inflation=lm.get("missing_variance_inflation", 2.0),
            )

        # Update ObservationModelConfig
        if "observation_model" in data:
            om = data["observation_model"]
            params.observation = ObservationModelConfig(
                base_reporting_prob=om.get("base_reporting_prob", 0.3),
                min_reporting_prob=om.get("min_reporting_prob", 0.01),
                max_reporting_prob=om.get("max_reporting_prob", 0.95),
                delay_mean_hours=om.get("delay_mean_hours", 2.0),
                delay_sd_hours=om.get("delay_sd_hours", 1.5),
                delay_max_hours=om.get("delay_max_hours", 48.0),
                rate_floor=om.get("rate_floor", 1e-6),
                coverage_threshold=om.get("coverage_threshold", 0.05),
            )

        # Update InferenceConfig
        if "inference" in data:
            i = data["inference"]
            params.inference = InferenceConfig(
                cred_interval=i.get("cred_interval", 0.9),
                min_event_duration_minutes=i.get("min_event_duration_minutes", 30),
                anomaly_zscore=i.get("anomaly_zscore", 2.0),
                posterior_variance_floor=i.get("posterior_variance_floor", 1e-6),
            )
            
        # Update ValidationConfig
        if "validation" in data:
            v = data["validation"]
            params.validation = ValidationConfig(
                rain_extreme_quantile=v.get("rain_extreme_quantile", 0.9),
                lead_time_threshold_minutes=v.get("lead_time_threshold_minutes", 120),
                stress_jump_zscore=v.get("stress_jump_zscore", 3.0),
                low_lying_quantile=v.get("low_lying_quantile", 0.2),
                overconfident_threshold=v.get("overconfident_threshold", 0.1),
                underconfident_threshold=v.get("underconfident_threshold", 0.5),
                stress_collapse_factor=v.get("stress_collapse_factor", 0.2),
                terrain_shuffle_drop=v.get("terrain_shuffle_drop", 0.2),
            )
            
        # Update OutputsConfig
        if "outputs" in data:
            o = data["outputs"]
            params.outputs = OutputsConfig(
                high_risk_threshold=o.get("high_risk_threshold", 0.8),
                medium_risk_threshold=o.get("medium_risk_threshold", 0.5),
                uncertainty_high=o.get("uncertainty_high", 0.4),
                uncertainty_low=o.get("uncertainty_low", 0.1),
                crs=o.get("crs", "EPSG:4326"),
                driver=o.get("driver", "GeoJSON"),
                html_max_points=o.get("html_max_points", 5000),
                snapshot_times=tuple(o.get("snapshot_times", ("pre", "peak", "post"))),
            )
            
    except Exception as e:
        # Log error to print but don't break
        print(f"Warning: Failed to load config from {config_path}: {e}")
        
    return params
