DataSetPath = './data/focusight1_round1_train_part1/OK_Images/'
DataSetPath_ = './data/focusight1_round1_train_part1/TC_images/'
ModelPath = './model/gen.ph'

lr = 0.003

epoch = 500

d_steps = 2

g_steps = 1

batch_size = 64

test_batch_size = 64

leakyrelu_alpha = 0.2

dropout_value = 0.25

save_single_image = True

save_grid_image = True