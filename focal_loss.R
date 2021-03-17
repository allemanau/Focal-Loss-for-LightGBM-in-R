focal_loss <- function(preds, dtrain, gamma = 1, alpha = .5){
    labels <- getinfo(dtrain, "label")
    sigmoid_pred <- 1 / (1 + exp(-preds))
    
    # gradient
    g1 <- sigmoid_pred * (1 - sigmoid_pred)
    g2 <- labels + ((-1)^labels) * sigmoid_pred
    g3 <- sigmoid_pred + labels - 1
    g4 <- 1 - labels - ((-1)^labels) * sigmoid_pred
    g5 <- labels + ((-1)^labels) * sigmoid_pred
    c1 <- 1 - labels - ((-1)^labels)*alpha
    c2 <- labels + ((-1)^labels)*alpha
    grad = gamma * c2 * g3 * g2^gamma * log(g4 + 1e-9) + ((-1)^labels) * c1 * g5^(gamma + 1)
    
    hess_1 = g2^gamma + gamma * ((-1)^labels) * g3 * g2^(gamma - 1)
    hess_2 = ((-1)^labels)*g3*g2^gamma/g4
    hess = ((hess_1*log(g4 + 1e-9) - hess_2)*c2*gamma + (gamma + 1)*c1*g5^gamma)*g1
    
    return(list(grad = grad, hess = hess))
}

focal_metric <- function(preds, dtrain, gamma = 1, alpha = .5){
    labels <- getinfo(dtrain, "label")
    preds[preds <= 0] <- 1e-9
    preds[preds >= 1] <- 1 - 1e-9
    # obj_value <- (-sum((1 - preds[labels == 1])^(gamma)*log(preds[labels == 1])) - 
    #                   sum((preds[labels == 0])^(gamma)*log(1 - preds[labels == 0])))/length(preds)
    obj_value <- (-sum(alpha*(1 - preds[labels == 1])^(gamma)*log(preds[labels == 1])) -
        sum((1-alpha)*(preds[labels == 0])^(gamma)*log(1 - preds[labels == 0])))/length(preds)
    return(list(name = "focal_loss", value = obj_value, higher_better = FALSE))
}
