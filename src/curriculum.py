import math


class Curriculum:
    """
    The class implements the different curriculum learning used to train the Transformers for linear regression.
    """
    def __init__(self, args):
        """
        Init of the class
        :param args: The specific parameters of the curriculum.
        """
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.dims.start  # dimension of the initial subspace in the prompts
        self.n_points = args.points.start  # number of points used initially in the prompts
        self.n_dims_schedule = args.dims  # member containing all the information regarding the dims: (dims.start,
                                          # dims.end, dims.interval and dims.inc)
        self.n_points_schedule = args.points  # same as args.dims but for the points
        self.step_count = 0  # keep a step count to update the parameters changing with curriculum
        self.sigma = args.sigma.start  # initial sigma for curriculum
        self.sigma_schedule = args.sigma  # all information regarding sigma

    def update(self):
        """
        Implements the update for curriculum learning, the function will be called at each training step.
        :return: None
        """
        self.step_count += 1  # increment by one the step count every training step

        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )  # update the number of dims
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)  # update number of points
        self.sigma = self.update_var(self.sigma, self.sigma_schedule)  # update sigma

    def update_var(self, var, schedule):
        """
        Increment the parameters interested by curriculum
        :param var: The current value of the param
        :param schedule: The member containing all the information such as (dims.start, dims.end, dims.interval and dims.inc)
        :return: The updated value of the param
        """
        # if it is time to increment (step count is divisible by the schedule) increment it
        if self.step_count % schedule.interval == 0:
            var += schedule.inc

        return min(var, schedule.end)  # to avoid to increase more than the endpoint


def get_final_var(init_var, total_steps, inc, n_steps, lim):
    """
    returns the final value of var after applying curriculum.
    :param init_var: The initial value of the param
    :param total_steps: The total number of steps
    :param inc: The increment in the parameter
    :param n_steps: The number of steps before an increment
    :param lim: The maximum number of the param
    :return: The value of the params after total_steps
    """
    final_var = init_var + math.floor(total_steps / n_steps) * inc

    return min(final_var, lim)
