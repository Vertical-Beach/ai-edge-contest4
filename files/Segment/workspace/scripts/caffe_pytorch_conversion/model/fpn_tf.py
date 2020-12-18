        conv1_7x7_s2 = FPN.conv(inputs, 7, 7, 64, 2, 2, relu=False, name='conv1_7x7_s2')
        conv1_7x7_s2_BatchNorm = FPN.batch_normalization(conv1_7x7_s2, scale_offset=False, relu=True, name='conv1_7x7_s2_BatchNorm',is_training = is_training)
        pool1_3x3_s2 = FPN.max_pool(conv1_7x7_s2_BatchNorm, 3, 3, 2, 2, name='pool1_3x3_s2')
        conv2_3x3_reduce = FPN.conv(pool1_3x3_s2, 1, 1, 64, 1, 1, relu=False, name='conv2_3x3_reduce')
        conv2_3x3_reduce_BatchNorm = FPN.batch_normalization(conv2_3x3_reduce, scale_offset=False, relu=True, name='conv2_3x3_reduce_BatchNorm',is_training = is_training)
        conv2_3x3 = FPN.conv(conv2_3x3_reduce_BatchNorm, 3, 3, 192, 1, 1, relu=False, name='conv2_3x3')
        conv2_3x3_BatchNorm = FPN.batch_normalization(conv2_3x3, scale_offset=False, relu=True, name='conv2_3x3_BatchNorm',is_training = is_training)
        pool2_3x3_s2 = FPN.max_pool(conv2_3x3_BatchNorm, 3, 3, 2, 2, name='pool2_3x3_s2')
        inception_3a_1x1 = FPN.conv(pool2_3x3_s2, 1, 1, 64, 1, 1, relu=False, name='inception_3a_1x1')
        inception_3a_1x1_BatchNorm = FPN.batch_normalization(inception_3a_1x1, scale_offset=False, relu=True, name='inception_3a_1x1_BatchNorm',is_training = is_training)

        inception_3a_3x3_reduce = FPN.conv(pool2_3x3_s2, 1, 1, 96, 1, 1, relu=False, name='inception_3a_3x3_reduce')
        inception_3a_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_3a_3x3_reduce, scale_offset=False, relu=True, name='inception_3a_3x3_reduce_BatchNorm',is_training = is_training)
        inception_3a_3x3 = FPN.conv(inception_3a_3x3_reduce_BatchNorm, 3, 3, 128, 1, 1, relu=False, name='inception_3a_3x3')
        inception_3a_3x3_BatchNorm = FPN.batch_normalization(inception_3a_3x3, scale_offset=False, relu=True, name='inception_3a_3x3_BatchNorm',is_training = is_training)

        inception_3a_5x5_reduce = FPN.conv(pool2_3x3_s2, 1, 1, 16, 1, 1, relu=False, name='inception_3a_5x5_reduce')
        inception_3a_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_3a_5x5_reduce, scale_offset=False, relu=True, name='inception_3a_5x5_reduce_BatchNorm',is_training = is_training)
        inception_3a_5x5 = FPN.conv(inception_3a_5x5_reduce_BatchNorm, 5, 5, 32, 1, 1, relu=False, name='inception_3a_5x5')
        inception_3a_5x5_BatchNorm = FPN.batch_normalization(inception_3a_5x5, scale_offset=False, relu=True, name='inception_3a_5x5_BatchNorm',is_training = is_training)

        inception_3a_pool = FPN.max_pool(pool2_3x3_s2, 3, 3, 1, 1, name='inception_3a_pool')
        inception_3a_pool_proj = FPN.conv(inception_3a_pool, 1, 1, 32, 1, 1, relu=False, name='inception_3a_pool_proj')
        inception_3a_pool_proj_BatchNorm = FPN.batch_normalization(inception_3a_pool_proj, scale_offset=False, relu=True, name='inception_3a_pool_proj_BatchNorm',is_training = is_training)

        inception_3a_output = FPN.concat([inception_3a_1x1_BatchNorm,inception_3a_3x3_BatchNorm,inception_3a_5x5_BatchNorm,inception_3a_pool_proj_BatchNorm], 3, name='inception_3a_output')
        inception_3b_1x1 = FPN.conv(inception_3a_output, 1, 1, 128, 1, 1, relu=False, name='inception_3b_1x1')
        inception_3b_1x1_BatchNorm = FPN.batch_normalization(inception_3b_1x1, scale_offset=False, relu=True, name='inception_3b_1x1_BatchNorm',is_training = is_training)

        inception_3b_3x3_reduce = FPN.conv(inception_3a_output, 1, 1, 128, 1, 1, relu=False, name='inception_3b_3x3_reduce')
        inception_3b_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_3b_3x3_reduce, scale_offset=False, relu=True, name='inception_3b_3x3_reduce_BatchNorm',is_training = is_training)
        inception_3b_3x3 = FPN.conv(inception_3b_3x3_reduce_BatchNorm, 3, 3, 192, 1, 1, relu=False, name='inception_3b_3x3')
        inception_3b_3x3_BatchNorm = FPN.batch_normalization(inception_3b_3x3, scale_offset=False, relu=True, name='inception_3b_3x3_BatchNorm',is_training = is_training)

        inception_3b_5x5_reduce = FPN.conv(inception_3a_output, 1, 1, 32, 1, 1, relu=False, name='inception_3b_5x5_reduce')
        inception_3b_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_3b_5x5_reduce, scale_offset=False, relu=True, name='inception_3b_5x5_reduce_BatchNorm',is_training = is_training)
        inception_3b_5x5 = FPN.conv(inception_3b_5x5_reduce_BatchNorm, 5, 5, 96, 1, 1, relu=False, name='inception_3b_5x5')
        inception_3b_5x5_BatchNorm = FPN.batch_normalization(inception_3b_5x5, scale_offset=False, relu=True, name='inception_3b_5x5_BatchNorm',is_training = is_training)

        inception_3b_pool = FPN.max_pool(inception_3a_output, 3, 3, 1, 1, name='inception_3b_pool')
        inception_3b_pool_proj = FPN.conv(inception_3b_pool, 1, 1, 64, 1, 1, relu=False, name='inception_3b_pool_proj')
        inception_3b_pool_proj_BatchNorm = FPN.batch_normalization(inception_3b_pool_proj, scale_offset=False, relu=True, name='inception_3b_pool_proj_BatchNorm',is_training = is_training)

        inception_3b_output = FPN.concat([inception_3b_1x1_BatchNorm,inception_3b_3x3_BatchNorm,inception_3b_5x5_BatchNorm,inception_3b_pool_proj_BatchNorm], 3, name='inception_3b_output')
        pool3_3x3_s2 = FPN.max_pool(inception_3b_output, 3, 3, 2, 2, name='pool3_3x3_s2')
        inception_4a_1x1 = FPN.conv(pool3_3x3_s2, 1, 1, 192, 1, 1, relu=False, name='inception_4a_1x1')
        inception_4a_1x1_BatchNorm = FPN.batch_normalization(inception_4a_1x1, scale_offset=False, relu=True, name='inception_4a_1x1_BatchNorm',is_training = is_training)

        inception_4a_3x3_reduce = FPN.conv(pool3_3x3_s2, 1, 1, 96, 1, 1, relu=False, name='inception_4a_3x3_reduce')
        inception_4a_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_4a_3x3_reduce, scale_offset=False, relu=True, name='inception_4a_3x3_reduce_BatchNorm',is_training = is_training)
        inception_4a_3x3 = FPN.conv(inception_4a_3x3_reduce_BatchNorm, 3, 3, 208, 1, 1, relu=False, name='inception_4a_3x3')
        inception_4a_3x3_BatchNorm = FPN.batch_normalization(inception_4a_3x3, scale_offset=False, relu=True, name='inception_4a_3x3_BatchNorm',is_training = is_training)

        inception_4a_5x5_reduce = FPN.conv(pool3_3x3_s2, 1, 1, 16, 1, 1, relu=False, name='inception_4a_5x5_reduce')
        inception_4a_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_4a_5x5_reduce, scale_offset=False, relu=True, name='inception_4a_5x5_reduce_BatchNorm',is_training = is_training)
        inception_4a_5x5 = FPN.conv(inception_4a_5x5_reduce_BatchNorm, 5, 5, 48, 1, 1, relu=False, name='inception_4a_5x5')
        inception_4a_5x5_BatchNorm = FPN.batch_normalization(inception_4a_5x5, scale_offset=False, relu=True, name='inception_4a_5x5_BatchNorm',is_training = is_training)

        inception_4a_pool = FPN.max_pool(pool3_3x3_s2, 3, 3, 1, 1, name='inception_4a_pool')
        inception_4a_pool_proj = FPN.conv(inception_4a_pool, 1, 1, 64, 1, 1, relu=False, name='inception_4a_pool_proj')
        inception_4a_pool_proj_BatchNorm = FPN.batch_normalization(inception_4a_pool_proj, scale_offset=False, relu=True, name='inception_4a_pool_proj_BatchNorm',is_training = is_training)

        inception_4a_output = FPN.concat([inception_4a_1x1_BatchNorm,inception_4a_3x3_BatchNorm,inception_4a_5x5_BatchNorm,inception_4a_pool_proj_BatchNorm], 3, name='inception_4a_output')
        inception_4b_1x1 = FPN.conv(inception_4a_output, 1, 1, 160, 1, 1, relu=False, name='inception_4b_1x1')
        inception_4b_1x1_BatchNorm = FPN.batch_normalization(inception_4b_1x1, scale_offset=False, relu=True, name='inception_4b_1x1_BatchNorm',is_training = is_training)

        inception_4b_3x3_reduce = FPN.conv(inception_4a_output, 1, 1, 112, 1, 1, relu=False, name='inception_4b_3x3_reduce')
        inception_4b_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_4b_3x3_reduce, scale_offset=False, relu=True, name='inception_4b_3x3_reduce_BatchNorm',is_training = is_training)
        inception_4b_3x3 = FPN.conv(inception_4b_3x3_reduce_BatchNorm, 3, 3, 224, 1, 1, relu=False, name='inception_4b_3x3')
        inception_4b_3x3_BatchNorm = FPN.batch_normalization(inception_4b_3x3, scale_offset=False, relu=True, name='inception_4b_3x3_BatchNorm',is_training = is_training)

        inception_4b_5x5_reduce = FPN.conv(inception_4a_output, 1, 1, 24, 1, 1, relu=False, name='inception_4b_5x5_reduce')
        inception_4b_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_4b_5x5_reduce, scale_offset=False, relu=True, name='inception_4b_5x5_reduce_BatchNorm',is_training = is_training)
        inception_4b_5x5 = FPN.conv(inception_4b_5x5_reduce_BatchNorm, 5, 5, 64, 1, 1, relu=False, name='inception_4b_5x5')
        inception_4b_5x5_BatchNorm = FPN.batch_normalization(inception_4b_5x5, scale_offset=False, relu=True, name='inception_4b_5x5_BatchNorm',is_training = is_training)

        inception_4b_pool = FPN.max_pool(inception_4a_output, 3, 3, 1, 1, name='inception_4b_pool')
        inception_4b_pool_proj = FPN.conv(inception_4b_pool, 1, 1, 64, 1, 1, relu=False, name='inception_4b_pool_proj')
        inception_4b_pool_proj_BatchNorm = FPN.batch_normalization(inception_4b_pool_proj, scale_offset=False, relu=True, name='inception_4b_pool_proj_BatchNorm',is_training = is_training)

        inception_4b_output = FPN.concat([inception_4b_1x1_BatchNorm,inception_4b_3x3_BatchNorm,inception_4b_5x5_BatchNorm,inception_4b_pool_proj_BatchNorm], 3, name='inception_4b_output')
        inception_4c_1x1 = FPN.conv(inception_4b_output, 1, 1, 128, 1, 1, relu=False, name='inception_4c_1x1')
        inception_4c_1x1_BatchNorm = FPN.batch_normalization(inception_4c_1x1, scale_offset=False, relu=True, name='inception_4c_1x1_BatchNorm',is_training = is_training)

        inception_4c_3x3_reduce = FPN.conv(inception_4b_output, 1, 1, 128, 1, 1, relu=False, name='inception_4c_3x3_reduce')
        inception_4c_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_4c_3x3_reduce, scale_offset=False, relu=True, name='inception_4c_3x3_reduce_BatchNorm',is_training = is_training)
        inception_4c_3x3 = FPN.conv(inception_4c_3x3_reduce_BatchNorm, 3, 3, 256, 1, 1, relu=False, name='inception_4c_3x3')
        inception_4c_3x3_BatchNorm = FPN.batch_normalization(inception_4c_3x3, scale_offset=False, relu=True, name='inception_4c_3x3_BatchNorm',is_training = is_training)

        inception_4c_5x5_reduce = FPN.conv(inception_4b_output, 1, 1, 24, 1, 1, relu=False, name='inception_4c_5x5_reduce')
        inception_4c_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_4c_5x5_reduce, scale_offset=False, relu=True, name='inception_4c_5x5_reduce_BatchNorm',is_training = is_training)
        inception_4c_5x5 = FPN.conv(inception_4c_5x5_reduce_BatchNorm, 5, 5, 64, 1, 1, relu=False, name='inception_4c_5x5')
        inception_4c_5x5_BatchNorm = FPN.batch_normalization(inception_4c_5x5, scale_offset=False, relu=True, name='inception_4c_5x5_BatchNorm',is_training = is_training)

        inception_4c_pool = FPN.max_pool(inception_4b_output, 3, 3, 1, 1, name='inception_4c_pool')
        inception_4c_pool_proj = FPN.conv(inception_4c_pool, 1, 1, 64, 1, 1, relu=False, name='inception_4c_pool_proj')
        inception_4c_pool_proj_BatchNorm = FPN.batch_normalization(inception_4c_pool_proj, scale_offset=False, relu=True, name='inception_4c_pool_proj_BatchNorm',is_training = is_training)

        inception_4c_output = FPN.concat([inception_4c_1x1_BatchNorm,inception_4c_3x3_BatchNorm,inception_4c_5x5_BatchNorm,inception_4c_pool_proj_BatchNorm], 3, name='inception_4c_output')
        inception_4d_1x1 = FPN.conv(inception_4c_output, 1, 1, 112, 1, 1, relu=False, name='inception_4d_1x1')
        inception_4d_1x1_BatchNorm = FPN.batch_normalization(inception_4d_1x1, scale_offset=False, relu=True, name='inception_4d_1x1_BatchNorm',is_training = is_training)

        inception_4d_3x3_reduce = FPN.conv(inception_4c_output, 1, 1, 144, 1, 1, relu=False, name='inception_4d_3x3_reduce')
        inception_4d_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_4d_3x3_reduce, scale_offset=False, relu=True, name='inception_4d_3x3_reduce_BatchNorm',is_training = is_training)
        inception_4d_3x3 = FPN.conv(inception_4d_3x3_reduce_BatchNorm, 3, 3, 288, 1, 1, relu=False, name='inception_4d_3x3')
        inception_4d_3x3_BatchNorm = FPN.batch_normalization(inception_4d_3x3, scale_offset=False, relu=True, name='inception_4d_3x3_BatchNorm',is_training = is_training)

        inception_4d_5x5_reduce = FPN.conv(inception_4c_output, 1, 1, 32, 1, 1, relu=False, name='inception_4d_5x5_reduce')
        inception_4d_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_4d_5x5_reduce, scale_offset=False, relu=True, name='inception_4d_5x5_reduce_BatchNorm',is_training = is_training)
        inception_4d_5x5 = FPN.conv(inception_4d_5x5_reduce_BatchNorm, 5, 5, 64, 1, 1, relu=False, name='inception_4d_5x5')
        inception_4d_5x5_BatchNorm = FPN.batch_normalization(inception_4d_5x5, scale_offset=False, relu=True, name='inception_4d_5x5_BatchNorm',is_training = is_training)

        inception_4d_pool = FPN.max_pool(inception_4c_output, 3, 3, 1, 1, name='inception_4d_pool')
        inception_4d_pool_proj = FPN.conv(inception_4d_pool, 1, 1, 64, 1, 1, relu=False, name='inception_4d_pool_proj')
        inception_4d_pool_proj_BatchNorm = FPN.batch_normalization(inception_4d_pool_proj, scale_offset=False, relu=True, name='inception_4d_pool_proj_BatchNorm',is_training = is_training)

        inception_4d_output = FPN.concat([inception_4d_1x1_BatchNorm,inception_4d_3x3_BatchNorm,inception_4d_5x5_BatchNorm,inception_4d_pool_proj_BatchNorm], 3, name='inception_4d_output')
        inception_4e_1x1 = FPN.conv(inception_4d_output, 1, 1, 256, 1, 1, relu=False, name='inception_4e_1x1')
        inception_4e_1x1_BatchNorm = FPN.batch_normalization(inception_4e_1x1, scale_offset=False, relu=True, name='inception_4e_1x1_BatchNorm',is_training = is_training)

        inception_4e_3x3_reduce = FPN.conv(inception_4d_output, 1, 1, 160, 1, 1, relu=False, name='inception_4e_3x3_reduce')
        inception_4e_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_4e_3x3_reduce, scale_offset=False, relu=True, name='inception_4e_3x3_reduce_BatchNorm',is_training = is_training)
        inception_4e_3x3 = FPN.conv(inception_4e_3x3_reduce_BatchNorm, 3, 3, 320, 1, 1, relu=False, name='inception_4e_3x3')
        inception_4e_3x3_BatchNorm = FPN.batch_normalization(inception_4e_3x3, scale_offset=False, relu=True, name='inception_4e_3x3_BatchNorm',is_training = is_training)

        inception_4e_5x5_reduce = FPN.conv(inception_4d_output, 1, 1, 32, 1, 1, relu=False, name='inception_4e_5x5_reduce')
        inception_4e_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_4e_5x5_reduce, scale_offset=False, relu=True, name='inception_4e_5x5_reduce_BatchNorm',is_training = is_training)
        inception_4e_5x5 = FPN.conv(inception_4e_5x5_reduce_BatchNorm, 5, 5, 128, 1, 1, relu=False, name='inception_4e_5x5')
        inception_4e_5x5_BatchNorm = FPN.batch_normalization(inception_4e_5x5, scale_offset=False, relu=True, name='inception_4e_5x5_BatchNorm',is_training = is_training)

        inception_4e_pool = FPN.max_pool(inception_4d_output, 3, 3, 1, 1, name='inception_4e_pool')
        inception_4e_pool_proj = FPN.conv(inception_4e_pool, 1, 1, 128, 1, 1, relu=False, name='inception_4e_pool_proj')
        inception_4e_pool_proj_BatchNorm = FPN.batch_normalization(inception_4e_pool_proj, scale_offset=False, relu=True, name='inception_4e_pool_proj_BatchNorm',is_training = is_training)

        inception_4e_output = FPN.concat([inception_4e_1x1_BatchNorm,inception_4e_3x3_BatchNorm,inception_4e_5x5_BatchNorm,inception_4e_pool_proj_BatchNorm], 3, name='inception_4e_output')
        pool4_3x3_s2 = FPN.max_pool(inception_4e_output, 3, 3, 2, 2, name='pool4_3x3_s2')
        inception_5a_1x1 = FPN.conv(pool4_3x3_s2, 1, 1, 256, 1, 1, relu=False, name='inception_5a_1x1')
        inception_5a_1x1_BatchNorm = FPN.batch_normalization(inception_5a_1x1, scale_offset=False, relu=True, name='inception_5a_1x1_BatchNorm',is_training = is_training)

        inception_5a_3x3_reduce = FPN.conv(pool4_3x3_s2, 1, 1, 160, 1, 1, relu=False, name='inception_5a_3x3_reduce')
        inception_5a_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_5a_3x3_reduce, scale_offset=False, relu=True, name='inception_5a_3x3_reduce_BatchNorm',is_training = is_training)
        inception_5a_3x3 = FPN.conv(inception_5a_3x3_reduce_BatchNorm, 3, 3, 320, 1, 1, relu=False, name='inception_5a_3x3')
        inception_5a_3x3_BatchNorm = FPN.batch_normalization(inception_5a_3x3, scale_offset=False, relu=True, name='inception_5a_3x3_BatchNorm',is_training = is_training)

        inception_5a_5x5_reduce = FPN.conv(pool4_3x3_s2, 1, 1, 32, 1, 1, relu=False, name='inception_5a_5x5_reduce')
        inception_5a_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_5a_5x5_reduce, scale_offset=False, relu=True, name='inception_5a_5x5_reduce_BatchNorm',is_training = is_training)
        inception_5a_5x5 = FPN.conv(inception_5a_5x5_reduce_BatchNorm, 5, 5, 128, 1, 1, relu=False, name='inception_5a_5x5')
        inception_5a_5x5_BatchNorm = FPN.batch_normalization(inception_5a_5x5, scale_offset=False, relu=True, name='inception_5a_5x5_BatchNorm',is_training = is_training)

        inception_5a_pool = FPN.max_pool(pool4_3x3_s2, 3, 3, 1, 1, name='inception_5a_pool')
        inception_5a_pool_proj = FPN.conv(inception_5a_pool, 1, 1, 128, 1, 1, relu=False, name='inception_5a_pool_proj')
        inception_5a_pool_proj_BatchNorm = FPN.batch_normalization(inception_5a_pool_proj, scale_offset=False, relu=True, name='inception_5a_pool_proj_BatchNorm',is_training = is_training)

        inception_5a_output = FPN.concat([inception_5a_1x1_BatchNorm,inception_5a_3x3_BatchNorm,inception_5a_5x5_BatchNorm,inception_5a_pool_proj_BatchNorm], 3, name='inception_5a_output')
        inception_5b_1x1 = FPN.conv(inception_5a_output, 1, 1, 384, 1, 1, relu=False, name='inception_5b_1x1')
        inception_5b_1x1_BatchNorm = FPN.batch_normalization(inception_5b_1x1, scale_offset=False, relu=True, name='inception_5b_1x1_BatchNorm',is_training = is_training)

        inception_5b_3x3_reduce = FPN.conv(inception_5a_output, 1, 1, 192, 1, 1, relu=False, name='inception_5b_3x3_reduce')
        inception_5b_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_5b_3x3_reduce, scale_offset=False, relu=True, name='inception_5b_3x3_reduce_BatchNorm',is_training = is_training)
        inception_5b_3x3 = FPN.conv(inception_5b_3x3_reduce_BatchNorm, 3, 3, 384, 1, 1, relu=False, name='inception_5b_3x3')
        inception_5b_3x3_BatchNorm = FPN.batch_normalization(inception_5b_3x3, scale_offset=False, relu=True, name='inception_5b_3x3_BatchNorm',is_training = is_training)

        inception_5b_5x5_reduce = FPN.conv(inception_5a_output, 1, 1, 48, 1, 1, relu=False, name='inception_5b_5x5_reduce')
        inception_5b_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_5b_5x5_reduce, scale_offset=False, relu=True, name='inception_5b_5x5_reduce_BatchNorm',is_training = is_training)
        inception_5b_5x5 = FPN.conv(inception_5b_5x5_reduce_BatchNorm, 5, 5, 128, 1, 1, relu=False, name='inception_5b_5x5')
        inception_5b_5x5_BatchNorm = FPN.batch_normalization(inception_5b_5x5, scale_offset=False, relu=True, name='inception_5b_5x5_BatchNorm',is_training = is_training)

        inception_5b_pool = FPN.max_pool(inception_5a_output, 3, 3, 1, 1, name='inception_5b_pool')
        inception_5b_pool_proj = FPN.conv(inception_5b_pool, 1, 1, 128, 1, 1, relu=False, name='inception_5b_pool_proj')
        inception_5b_pool_proj_BatchNorm = FPN.batch_normalization(inception_5b_pool_proj, scale_offset=False, relu=True, name='inception_5b_pool_proj_BatchNorm',is_training = is_training)

        inception_5b_output = FPN.concat([inception_5b_1x1_BatchNorm,inception_5b_3x3_BatchNorm,inception_5b_5x5_BatchNorm,inception_5b_pool_proj_BatchNorm], 3, name='inception_5b_output')
        p5 = FPN.conv(inception_5b_output, 1, 1, 32, 1, 1, relu=False, name='p5')
        upsample_p5 = FPN.deconv(p5, 4, 4, 32, 2, 2, padding=1, biased=False, relu=False, name='upsample_p5')

        latlayer_4f = FPN.conv(inception_4e_output, 1, 1, 32, 1, 1, relu=False, name='latlayer_4f')

        add_p4 = FPN.add([upsample_p5,latlayer_4f], name='add_p4')
        toplayer_p4 = FPN.conv(add_p4, 3, 3, 32, 1, 1, relu=False, name='toplayer_p4')
        upsample_p4 = FPN.deconv(toplayer_p4, 4, 4, 32, 2, 2, padding=1, biased=False, relu=False, name='upsample_p4')
        latlayer_3d = FPN.conv(inception_3b_output, 1, 1, 32, 1, 1, relu=False, name='latlayer_3d')

        add_p3 = FPN.add([upsample_p4,latlayer_3d], name='add_p3')
        toplayer_p3 = FPN.conv(add_p3, 3, 3, 32, 1, 1, relu=False, name='toplayer_p3')
        upsample_p3 = FPN.deconv(toplayer_p3, 4, 4, 32, 2, 2, padding=1, biased=False, relu=False, name='upsample_p3')

        latlayer_2c = FPN.conv(conv2_3x3_BatchNorm, 1, 1, 32, 1, 1, relu=False, name='latlayer_2c')

        add_p2 = FPN.add([upsample_p3,latlayer_2c], name='add_p2')
        toplayer_p2 = FPN.deconv(add_p2, 4, 4, num_classes, 4, 4, padding=0, relu=False, name='toplayer_p2')