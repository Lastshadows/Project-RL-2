import torch
from domain import Domain

def PQL(trajectory, learning_rate, gamma):
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N = len(trajectory)
    D_in = 3
    H = 100
    D_out = 1

    domain = Domain()

    # Create Tensor holding our data

    x2 = torch.randn(N, D_in)
    y2 = torch.randn(N, D_out)

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
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )

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

        y_pred2 = model(x2)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y)
        loss2 = loss_fn(y_pred2, y2)

        print(" LOSS = " + str(loss))
        print(" LOSS2 = " + str(loss2))
        if i % 100 == 99:
            print(i, loss2.item())

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.

        loss2.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                print(" \n !!! learning rate : " + str(learning_rate))
                print(" !!! param.grid : " + str(param.grad))
                print(" !!! temporal_delta : " + str(temporal_delta))
                param += learning_rate * param.grad * temporal_delta

        if i > 4:
            break
