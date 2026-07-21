# Data Recording


Sometimes it is useful to store the snapshots of data during the calculation for some tasks such as debugging. QURI Parts provides the smart way to do it with `@recordable` decorator.

Here we create a function that calculates the expectation value of given operators with given quantum state and returns the highest value. Using recorder feature you can store all estimates and retrieve the data outside of the function.

## Setup 
As an example for demonstration, we first create 10 random operators and a random quantum state. Then we implement a recordable function whose purpose is to estimate the expectation value of the 10 operators on the random state and return the maximal estimate.


```python
import random
import numpy as np

from quri_parts.circuit import QuantumCircuit
from quri_parts.core.operator import Operator, PauliLabel
from quri_parts.core.state.state_helper import quantum_state


qubit_count = 5

# create random state
circuit = QuantumCircuit(qubit_count)
for i in range(qubit_count):
    circuit.add_RX_gate(i, random.random() * 2 * np.pi)
    circuit.add_RY_gate(i, random.random() * 2 * np.pi)
state = quantum_state(qubit_count, circuit=circuit)

# create 10 random operators
ops = []
for _ in range(10):
    coef = random.random()
    p_ids = []
    p_indices = []
    for i in range(qubit_count):
        id = random.randint(0, 3)
        if id != 0:
            p_ids.append(id)
            p_indices.append(i)
    ops.append(Operator({PauliLabel.from_index_and_pauli_list(p_indices, p_ids): coef}))
```

## Implement and run the recordable function

Here, we implement the `return_highest_estimate` function that iterates over a list of random `Operator`s and a quantum state for estimation.


```python
from quri_parts.core.utils.recording import Recorder, recordable
from quri_parts.qulacs.estimator import create_qulacs_vector_estimator
from quri_parts.core.state import CircuitQuantumState

# This function returns the highest expectation value of the given operators
# Add @recordable decorator to the function to make it recordable
@recordable
def return_highest_estimate(
    recorder: Recorder, ops: list[Operator], state: CircuitQuantumState
) -> float:
    estimator = create_qulacs_vector_estimator()
    est_vals = []
    for i, op in enumerate(ops):
        est_vals.append(estimator(op, state).value.real)
        recorder.info("expectation value {}".format(i), est_vals[i])
        recorder.debug("operator", op)
    return max(est_vals)
```

Note that a recordable function is supposed to take a `Recorder` as its first argument. However, with the help of the `@recordable` decorator, you can ignore the `recorder` argument when you just want to call it like a regular function. In this case, nothing will be reocorded.


```python
return_highest_estimate(ops, state)
```



```output
0.1021413350439432
```


Calling it like a usual function does not record anything. To activate recording, you would need to create a `RecordingSession`. Afterwards, we need to set the recording level the function. We provide 2 recording levels `INFO` and `DEBUG`.

- When logging level is `INFO`, only data recorded by `.info` in a recordable function will be recorded.
- When logging level is `DEBUG`, data recorded by `.info` and `.debug` in a recordable function will both be recorded.

With this in mind, let's present how to perform recording.


```python
from quri_parts.core.utils.recording import INFO, RecordSession

# Create a record session and set the record level
session = RecordSession()
session.set_level(INFO, return_highest_estimate)

# Create a record context and run the function in it
with session.start():
    # You don't need to pass the recorder to the function when you call it
    highest = return_highest_estimate(ops, state)
    print(f"Highest estimate: {highest}")
```
```output
Highest estimate: 0.1021413350439432
```
Finally, let's retrieve the data from the record session. Each call to a recordable function creates a `RecordGroup`. A `RecordGroup` contains all the values recodered inside a recordable function. In this case, `return_highest_estimate` is only called once in the session. Thus only 1 group is present. The data is retrieved by:


```python
rec_group = list(session.get_records().get_history(return_highest_estimate))[0]
print(rec_group)

estimates = [entry.data[1] for entry in rec_group.entries]
print("Estimates:")
print(sorted(estimates))
```
```output
RecordGroup(
  func_id: __main__.return_highest_estimate,
  entries: [
    INFO:__main__.return_highest_estimate:('expectation value 0', 0.005362605340838037),
    INFO:__main__.return_highest_estimate:('expectation value 1', -0.014474899729036051),
    INFO:__main__.return_highest_estimate:('expectation value 2', -0.8579842355246938),
    INFO:__main__.return_highest_estimate:('expectation value 3', -0.009675019926336322),
    INFO:__main__.return_highest_estimate:('expectation value 4', 0.1021413350439432),
    INFO:__main__.return_highest_estimate:('expectation value 5', -0.052663804784036324),
    INFO:__main__.return_highest_estimate:('expectation value 6', 0.0006976455786803415),
    INFO:__main__.return_highest_estimate:('expectation value 7', 0.028763410883260798),
    INFO:__main__.return_highest_estimate:('expectation value 8', -0.00943261636079905),
    INFO:__main__.return_highest_estimate:('expectation value 9', -0.035973003036850394),
  ]
)
Estimates:
[-0.8579842355246938, -0.052663804784036324, -0.035973003036850394, -0.014474899729036051, -0.009675019926336322, -0.00943261636079905, 0.0006976455786803415, 0.005362605340838037, 0.028763410883260798, 0.1021413350439432]
```
For completeness, we also show what happens if the record level is set to `DEBUG`.


```python
from quri_parts.core.utils.recording import DEBUG

# Create a record session and set the record level
debug_session = RecordSession()
debug_session.set_level(DEBUG, return_highest_estimate)

# Create a record context and run the function in it
with debug_session.start():
    # You don't need to pass the recorder to the function when you call it
    highest = return_highest_estimate(ops, state)


print(
    next(debug_session.get_records().get_history(return_highest_estimate))
)
```
```output
RecordGroup(
  func_id: __main__.return_highest_estimate,
  entries: [
    INFO:__main__.return_highest_estimate:('expectation value 0', 0.005362605340838037),
    DEBUG:__main__.return_highest_estimate:('operator', {PauliLabel({(1, 1), (0, 3), (4, 1), (3, 3)}): 0.16992698807498363}),
    INFO:__main__.return_highest_estimate:('expectation value 1', -0.014474899729036051),
    DEBUG:__main__.return_highest_estimate:('operator', {PauliLabel({(2, 3), (1, 1), (0, 3), (3, 1)}): 0.12553815100764432}),
    INFO:__main__.return_highest_estimate:('expectation value 2', -0.8579842355246938),
    DEBUG:__main__.return_highest_estimate:('operator', {PauliLabel({(4, 2)}): 0.8704939578861669}),
    INFO:__main__.return_highest_estimate:('expectation value 3', -0.009675019926336322),
    DEBUG:__main__.return_highest_estimate:('operator', {PauliLabel({(0, 1), (1, 2), (2, 1)}): 0.5777420680384585}),
    INFO:__main__.return_highest_estimate:('expectation value 4', 0.1021413350439432),
    DEBUG:__main__.return_highest_estimate:('operator', {PauliLabel({(1, 2), (3, 3), (4, 2)}): 0.24594316144478523}),
    INFO:__main__.return_highest_estimate:('expectation value 5', -0.052663804784036324),
    DEBUG:__main__.return_highest_estimate:('operator', {PauliLabel({(0, 1), (1, 1), (3, 2)}): 0.17277183082697856}),
    INFO:__main__.return_highest_estimate:('expectation value 6', 0.0006976455786803415),
    DEBUG:__main__.return_highest_estimate:('operator', {PauliLabel({(2, 3), (3, 3), (1, 3), (4, 1)}): 0.26586144825708646}),
    INFO:__main__.return_highest_estimate:('expectation value 7', 0.028763410883260798),
    DEBUG:__main__.return_highest_estimate:('operator', {PauliLabel({(3, 1), (1, 1), (4, 2), (2, 3), (0, 2)}): 0.3788376395360946}),
    INFO:__main__.return_highest_estimate:('expectation value 8', -0.00943261636079905),
    DEBUG:__main__.return_highest_estimate:('operator', {PauliLabel({(0, 1), (4, 3), (3, 1), (2, 2)}): 0.9499744000925475}),
    INFO:__main__.return_highest_estimate:('expectation value 9', -0.035973003036850394),
    DEBUG:__main__.return_highest_estimate:('operator', {PauliLabel({(1, 1), (0, 2), (2, 3)}): 0.1165934189303991}),
  ]
)
```