import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def Least_squares(x,y):
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(len(x)):
        k = (x[i] - x_) * (y[i] - y_)
        m += k
        p = np.square(x[i] - x_)
        n = n + p
    a = m / n
    b = y_ - a * x_
    return a, b

if __name__ == '__main__':
    # load data
    poses = np.loadtxt("poses.txt")
    timestamps = (np.loadtxt("times.txt").reshape((-1, 1)))[0:100]
    # extract x, y, z
    trajectory_history = (poses.T[[3, 7, 11]]).T[0:100]  #x, y, z : x: right; y: down; z: forward.
    trajectory_history = np.c_[trajectory_history, timestamps] # [x, y, z, timestamp]

    time_interval = trajectory_history[:, 3]
    totalframe_num = trajectory_history.shape[0]
    # use past k frames to predict
    k = 10
    predict_result = np.zeros((trajectory_history.shape[0]-k, trajectory_history.shape[1]))
    predict_parameter = np.zeros((totalframe_num - k, 7))

    for i in range(totalframe_num - k):
        use_time_interval = time_interval[i: i + k - 1]
        use_time_interval = use_time_interval - use_time_interval[0]
        x_history = trajectory_history[i:i+k-1, 0]
        y_history = trajectory_history[i:i+k-1, 1]
        z_history = trajectory_history[i:i+k-1, 2]
        vx, x0 = Least_squares(use_time_interval, x_history)
        vy, y0 = Least_squares(use_time_interval, y_history)
        vz, z0 = Least_squares(use_time_interval, z_history)
        predict_parameter[i] = [vx, x0, vy, y0, vz, z0, use_time_interval[-1] + time_interval[i] + 0.5]

        # the reaction time of normal people is 0.2s-0.3s, here we leave some margin and suppose the reaction time is 0.5s
        # so at each timestamp tnow, we use the trajectory history of a vehicle from the past k frames until now, and predict
        # each vehicle's position at time tnow + 0.5s.
        x_predict = (use_time_interval[-1] + 0.5) * vx + x0
        y_predict = (use_time_interval[-1] + 0.5) * vy + y0
        z_predict = (use_time_interval[-1] + 0.5) * vz + z0
        predict_result[i] = [x_predict, y_predict, z_predict, use_time_interval[-1] + time_interval[i] + 0.5]

    # draw the groundtruth trajectory and the trajectory we predict.
    # draw the result of x axis
    plt.plot(time_interval, trajectory_history[:, 0], 's-', color = 'r', label='trajectory truth')
    plt.plot(predict_result[:, 3], predict_result[:, 0], 'o-', color = 'b', label='predicted trajectory')
    plt.title('Predicted x values and the groundtruth')
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.legend(loc = "best")
    plt.savefig('./x.jpg')
    plt.show()
    # draw the result of y axis
    plt.plot(time_interval, trajectory_history[:, 1], 's-', color = 'r', label='trajectory truth')
    plt.plot(predict_result[:, 3], predict_result[:, 1], 'o-', color = 'b', label='predicted trajectory')
    plt.title('Predicted y values and the groundtruth')
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.legend(loc="best")
    plt.savefig('./y.jpg')
    plt.show()
    # draw the result of z axis
    plt.plot(time_interval, trajectory_history[:, 2], 's-', color = 'r', label='trajectory truth')
    plt.plot(predict_result[:, 3], predict_result[:, 2], 'o-', color = 'b', label='predicted trajectory')
    plt.title('Predicted z values and the groundtruth')
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.legend(loc="best")
    plt.savefig('./z.jpg')
    plt.show()
    # draw the result of (x,z)
    plt.plot(trajectory_history[:, 0], trajectory_history[:, 2], 's-', color = 'r', label='trajectory truth')
    plt.plot(predict_result[:, 0], predict_result[:, 2], 'o-', color = 'b', label='predicted trajectory')
    plt.title('Predicted trajectory (x,z) and the groundtruth')
    plt.xlabel("x(m)")
    plt.ylabel("z(m)")
    plt.legend(loc="best")
    plt.savefig('./xz_plane.jpg')
    plt.show()

    # calculate the predicting error
    for i in range(totalframe_num):
        if trajectory_history[i,3] >= time_interval[k-1] + 0.5:
            begin_point = i
            break
    error = np.zeros((totalframe_num - begin_point, 3)) # error_x, error_y, error_z

    for i in range(begin_point, totalframe_num):
        for j in range(predict_parameter.shape[0]):
            if trajectory_history[i,3] >= predict_parameter[j,6] and trajectory_history[i,3] < predict_parameter[j + 1, 6]:
                predict_value_x = predict_parameter[j, 0] * (trajectory_history[i, 3] - time_interval[j]) + \
                                  predict_parameter[j, 1]
                predict_value_y = predict_parameter[j, 2] * (trajectory_history[i, 3] - time_interval[j]) + \
                                  predict_parameter[j, 3]
                predict_value_z = predict_parameter[j, 4] * (trajectory_history[i, 3] - time_interval[j]) + \
                                  predict_parameter[j, 5]
                error_x = abs(trajectory_history[i, 0] - predict_value_x)
                error_y = abs(trajectory_history[i, 1] - predict_value_y)
                error_z = abs(trajectory_history[i, 2] - predict_value_z)
                error[i - begin_point] = [error_x, error_y, error_z]
                break
                print(error)

    # mean error in x axis
    meanerror_x = (np.sum(error[:, 0])) / error.shape[0]
    # mean error in y axis
    meanerror_y = (np.sum(error[:, 1])) / error.shape[0]
    # mean error in z axis
    meanerror_z = (np.sum(error[:, 2])) / error.shape[0]
    # RMSE in (x,z) plane
    xy_rmse = sqrt((np.sum(error[:, 0] ** 2) + np.sum(error[:, 2] ** 2)) / error.shape[0])
    print(meanerror_x, meanerror_y, meanerror_z, xy_rmse)


