import torch
from domain import Domain

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.1)

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

def PQL(trajectory, learning_rate, gamma):

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N = len(trajectory)
    D_in = 3
    H1 = 50
    H2 = 100
    H3 = 50
    D_out = 1

    domain = Domain()

    # Create Tensor holding our data
    x = []
    y = []

    for tuple in trajectory:
        ((p, s), action,(p2,s2), r) =  tuple
        x.append([p,s,action])
        y.append([r])


    x = torch.tensor(x)
    y = torch.tensor(y)

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H1, H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H1, H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H1, H2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H2, H2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H2, H3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H3, H2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H2, H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H1, H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H1, H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H1, H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H1, H2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H2, H2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H2, H3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H3, H2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H2, H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H1, H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H1, D_out),
    )

    # initialize weights
    #model.apply(init_weights)
    model.apply(weights_init_uniform)

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(reduction='sum')

    for i in range(len(trajectory)):
        print("\n NEW ITERATION")
        print(i)
        # creating the temporal difference factor

        # computing the max Q from possible actions
        maxQ = -10
        for action in domain.ACTIONS:
            p,s = trajectory[i][2]
            Q = [p,s,action]
            r = model(torch.tensor(Q))

            if maxQ<r:
                maxQ = r

        p =  x[i][0]
        s =  x[i][1]
        u =  x[i][2]
        Q = [p,s,u]
        temporal_delta =  y[i] + gamma*maxQ - model(torch.tensor(Q))

        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y.float())

        print(" LOSS = " + str(loss) )

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.

        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                """
                print(" \n !!! learning rate : " + str(learning_rate))
                print(" !!! param.grid : " + str(param.grad))
                print(" !!! temporal_delta : " + str(temporal_delta))
                """
                param += learning_rate * param.grad * temporal_delta
