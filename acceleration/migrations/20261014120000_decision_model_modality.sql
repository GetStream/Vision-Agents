-- +goose Up

-- The lcm modality is called decision_model now. Stats keep reporting under one name.
UPDATE requests SET modality = 'decision_model' WHERE modality = 'lcm';
UPDATE stats_hourly SET modality = 'decision_model' WHERE modality = 'lcm';
UPDATE stats_daily SET modality = 'decision_model' WHERE modality = 'lcm';
UPDATE stats_tags_hourly SET modality = 'decision_model' WHERE modality = 'lcm';
UPDATE stats_tags_daily SET modality = 'decision_model' WHERE modality = 'lcm';

-- +goose Down
UPDATE stats_tags_daily SET modality = 'lcm' WHERE modality = 'decision_model';
UPDATE stats_tags_hourly SET modality = 'lcm' WHERE modality = 'decision_model';
UPDATE stats_daily SET modality = 'lcm' WHERE modality = 'decision_model';
UPDATE stats_hourly SET modality = 'lcm' WHERE modality = 'decision_model';
UPDATE requests SET modality = 'lcm' WHERE modality = 'decision_model';
