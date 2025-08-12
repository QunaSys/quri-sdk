using ITensors
function init_state(s)
    psi = productMPS(s, ["0", "0", "1", "0", "0", "0"])
    return psi
end
