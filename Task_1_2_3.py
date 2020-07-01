import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as interpolate


def linear_regression(X, f_x):
    # we can rewrite the line equation as y = Ap, where A=[[x, 1]] and p=[[m], [c]]

    if len(X.shape) is 1:
        A = np.c_[X, np.ones(len(X))]
    else:
        A = X
    p, residuals, rank, svals = np.linalg.lstsq(A, f_x)
    # solving for p gives us the slope and intercept
    return A, p


def calculate_plot_linear_regression(X, f_x):
    A, p = linear_regression(X, f_x)

    fig, ax = plt.subplots(1, 1)
    ax.plot(X, f_x, 'o', label='Data')
    ax.plot(X, A.dot(p), '-k', color='orange', lw=2, label='Linear fit')
    ax.legend()

    fig.show()
    fig.savefig('t1_linear.png')


def get_centroids(X, n_centroids):
    # Find the indeces
    idx = np.random.randint(np.size(X, axis=0), size=n_centroids)
    # Use indeces to grab rows
    return idx, X[idx]


def calculate_distance(x, cent_i):
    ## calculate the distance between the centroid and the other elements

    if len(x.shape) is 1:
        return (x - cent_i) ** 2
    else:
        return np.sum((x - cent_i) ** 2, axis=1)


def non_linear_regression(X, f_x, n_centroids, bandwidth):
    n = len(X)

    idx, centroids = get_centroids(X, n_centroids)

    phi_train = np.ones((n, 1))

    for i in range(n_centroids):
        cent_i = centroids[i]  # get the ith centroid
        dist_i = calculate_distance(X, cent_i)  ## distance matrix for i th centroid
        phi_i = np.exp(-dist_i / (bandwidth ** 2))  ##
        phi_i = np.reshape(phi_i, (n, 1))  # dummy rehsape to (n,1)
        phi_train = np.hstack((phi_train, phi_i))

    A, p = linear_regression(phi_train, f_x)

    return A, p


def calculate_plot_non_linear_regression(X, f_x, n_centroids, bandwidth):
    A, p = non_linear_regression(X, f_x, n_centroids, bandwidth)

    fig, ax = plt.subplots(1, 1)
    ax.plot(X, f_x, 'o', label='Data')
    ax.plot(X, A.dot(p), 'x', color='orange', label='Non-linear fit')
    ax.legend()

    fig.show()
    fig.savefig('t1_non_linear_1_30.png')


def estimate_vector_field(X_0, X_1, delta):
    return (X_1 - X_0) / delta


def calculate_plot_linear_vec_field(X_0, X_1):
    ###vector field linear regression
    d_t = 0.1

    v_head = estimate_vector_field(X_0, X_1, d_t)

    A, p = linear_regression(X_0, v_head)

    X_head = p @ X_0.T
    X_head = X_head.T

    error = np.sum((X_head - X_1) ** 2) / len(X_0)

    print(f' Squared error = {error}')

    ## calculate and plot trajectory from 10,10

    number_steps = 1000

    x_s = np.empty(number_steps + 1)
    y_s = np.empty(number_steps + 1)

    x_s[0] = 10
    y_s[0] = 10

    for i in range(number_steps):
        x_s[i + 1] = x_s[i] + ((x_s[i] * p[0][0] + y_s[i] * p[0][1]) * d_t)

        y_s[i + 1] = y_s[i] + ((y_s[i] * p[1][0] + y_s[i] * p[1][1]) * d_t)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_s, y_s, lw=3)

    ### calculate and plot the phase diagram
    X, Y = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))

    u, v = np.zeros(X.shape), np.zeros(Y.shape)

    NI, NJ = X.shape

    for i in range(NI):
        for j in range(NJ):
            x = X[i, j]
            y = Y[i, j]

            u[i, j] = x * p[0][0] + y * p[0][1]
            v[i, j] = x * p[1][0] + y * p[1][1]

    ax.streamplot(X, Y, u, v, color=u ** 2 + v ** 2, cmap=plt.cm.autumn)

    fig.show()
    fig.savefig('t2_linear_vector.png')

def calculate_plot_nonlinear_vec_field(X_0, X_1):
    ###vector field linear regression
    d_t = 0.15

    v_head = estimate_vector_field(X_0, X_1, d_t)

    A, p = linear_regression(X_0, v_head)

    X_head = p @ X_0.T
    X_head = X_head.T

    error = np.sum((X_head - X_1) ** 2) / len(X_0)

    print(f'Linear Squared error = {error}')

    ## calculate and plot trajectory from 4,4

    number_steps = 100

    x_s = np.empty(number_steps + 1)
    y_s = np.empty(number_steps + 1)

    x_s[0] = 4
    y_s[0] = 4

    for i in range(number_steps):
        x_s[i + 1] = x_s[i] + ((x_s[i] * p[0][0] + y_s[i] * p[0][1]) * d_t)

        y_s[i + 1] = y_s[i] + ((y_s[i] * p[1][0] + y_s[i] * p[1][1]) * d_t)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_s, y_s, lw=3)

    ### calculate and plot the phase diagram
    X, Y = np.meshgrid(np.linspace(-4, 4, 500), np.linspace(-4, 4, 500))

    u, v = np.zeros(X.shape), np.zeros(Y.shape)

    NI, NJ = X.shape

    for i in range(NI):
        for j in range(NJ):
            x = X[i, j]
            y = Y[i, j]

            u[i, j] = x * p[0][0] + y * p[0][1]
            v[i, j] = x * p[1][0] + y * p[1][1]

    ax.streamplot(X, Y, u, v, color=u ** 2 + v ** 2, cmap=plt.cm.autumn)

    fig.show()
    fig.savefig('t3_linear_vector.png')

    ### estimate and do it again for the non linear regression

    n_centroids = 100
    bandwidth = 0.1

    A, p = non_linear_regression(X_0, v_head, n_centroids, bandwidth)

    X_head= A @ p

    X_head = X_0 + X_head

    error = np.sum((X_head - X_1) ** 2) / len(X_0)

    print(f'Non-linear squared error = {error}')


    x = X_0[:, 0]
    y = X_0[:, 1]
    u = X_head[:, 0]
    v = X_head[:, 1]

    xi = np.linspace(x.min(), x.max(), 500)
    yi = np.linspace(y.min(), y.max(), 500)
    X, Y = np.meshgrid(xi, yi)
    U = interpolate.griddata((x, y), u, (X, Y), method='cubic')
    V = interpolate.griddata((x, y), v, (X, Y), method='cubic')

    plt.figure()
    plt.streamplot(X, Y, -U, -V, color=U ** 2 + V ** 2, cmap=plt.cm.autumn)

    plt.savefig('t3_non_linear_vector.png')
    plt.show()




######  Task 1 - 1  ###########
data = np.loadtxt('linear_function_data.txt')

X = data[:, 0]
f_x = data[:, 1]

##calculate_plot_linear_regression(X, f_x)

###### Task 1 - 2  ###########
data = np.loadtxt('nonlinear_function_data.txt')

X = data[:, 0]
f_x = data[:, 1]

calculate_plot_linear_regression(X, f_x)

###### Task 1 - 3  ##########
data = np.loadtxt('nonlinear_function_data.txt')

n_centroids = 30

X = data[:, 0]
f_x = data[:, 1]

bandwidth = 0.5

calculate_plot_non_linear_regression(X, f_x, n_centroids, bandwidth)

###########################################

#######  Task 2   ###########
X_0 = np.loadtxt('linear_vectorfield_data_x0.txt')
X_1 = np.loadtxt('linear_vectorfield_data_x1.txt')

## first plot original data
fig, ax = plt.subplots(1, 1)
ax.scatter(X_0[:, 0], X_0[:, 1],  label='X_0', s=5)
ax.scatter(X_1[:, 0], X_1[:, 1], color='orange', label='X_1', s=5)
ax.legend()
fig.show()
fig.savefig('t2_original_data.png')

calculate_plot_linear_vec_field(X_0, X_1)

#####################################

######### Task 3 ###########

X_0 = np.loadtxt('nonlinear_vectorfield_data_x0.txt')
X_1 = np.loadtxt('nonlinear_vectorfield_data_x1.txt')

## first plot original data
fig, ax = plt.subplots(1, 1)
ax.scatter(X_0[:, 0], X_0[:, 1],  label='X_0', s=5)
ax.scatter(X_1[:, 0], X_1[:, 1], color='orange', label='X_1', s=5)
ax.legend()
fig.show()
fig.savefig('t3_original_data.png')

calculate_plot_nonlinear_vec_field(X_0, X_1)
