from runner import wgan_gp_run

wgan_gp_run(train_window = 52,
            test_size = 52,
            horizon = 26,
            batch_size = 6,
            epochs = 25000,
            d_iterations = 3,
            time_series = True,
            BERT = True, 
            load = None,
            cuda_no = 'cuda:0')
