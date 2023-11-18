# method = 'SimVP'
# model_type = 'convnext'
# hid_S = 64
# hid_T = 512
# N_T = 10
# N_S = 4

method = 'SimVP'
model_type = 'slanet'
hid_S = 64
hid_T = 128
N_T = 2
N_S = 2
lr = 1e-2
batch_size = 16
drop_path = 0.1
sched = 'onecycle'
warmup_epoch = 5