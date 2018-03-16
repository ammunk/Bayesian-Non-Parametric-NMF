import time
import os
import logging
import pickle


class Logger:

    def __init__(self, user_params):

        logging.basicConfig(format = ('%(asctime)s, %(levelname)s, '
                                      '%(name)s, -- %(message)s'),
                        datefmt = "%m-%d %H:%M:%S",
                        level = logging.DEBUG)
        self.user_params = user_params
        timer = time.time()
	# time.gmtime converts seconds to correct format for strftime.
        timer = time.strftime("%m-%d-%H-%M-%S", time.gmtime(timer))

        self.data_file = os.path.abspath(
                    os.path.join(os.path.dirname( __file__ ), '..', 
                                'results', timer) + '.pickle')

        self.it = 0
        self.data = {'elbo': [], 'sq_error': []}
        self.on_hpc = user_params['on_hpc']

    def __enter__(self):

        logging.info(("\nPerforming non-negative matrix factorization (NMF) "
                      "using following setings:\n\n"
                            "\t - n_split: {n_split}\n"
                            "\t - inference type: {inference_type}\n"
                            "\t - init_type: {init_type}\n"
                            "\t - Data used: {use_data}\n"
                            "\t - Maximum iterations: {max_iter}\n"
                            "\t - Tolerance: {tolerance}\n")
                            .format(**self.user_params))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.data, f)

    def log_update(self, elbo, iteration, sq_error, elbo_diff):
        self.it += 1
        if self.it == 1 and not self.on_hpc:
            msg = (f"\n\tIteration = {iteration}\n\tELBO diff = {elbo_diff}\n"
                  f"\tSq Error = {sq_error}\n================================"
                  "==============")
            logging.info(msg)
            self.it = 0
        self.data['elbo'].append(elbo)
        self.data['sq_error'].append(sq_error)
