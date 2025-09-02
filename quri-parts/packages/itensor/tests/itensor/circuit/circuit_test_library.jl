using ITensors

# return state Vector of the MPS. (<psi|0>, <psi|1>, ..., <psi|2^length(s)-1>)
function stateVector(psi, s)
    stateVector = []
    for i in 1:(2^length(s))
        push!(stateVector, inner(productMPS(s,reverse([string(string(i-1, base=2, pad=length(s))[j]) for j in 1:length(s)])), psi))
    end
    return stateVector
end
