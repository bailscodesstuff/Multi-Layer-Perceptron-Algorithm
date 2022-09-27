# generating getters and setters for the nodes based on the attributes.
# I believe the code is self-explanatory.

# node superclass
class Node:
    def __init__(self,weight,activation,delta):
        self.weight = weight
        self.activation = activation
        self.delta = delta

    # @property is how getters and setters are generated with python
    @property
    def weight(self):
        return self._weight
    # @attribute.setter is how setters are generated with python
    @weight.setter
    def weight(self,weight):
        self._weight = weight
    @property
    def activation(self):
        return self._activation
    @activation.setter
    def activation(self,activation):
        self._activation = activation
    @property
    def delta(self):
        return self._delta
    @delta.setter
    def delta(self,delta):
        self._delta = delta



class inputNode(Node):
    def __init__(self,weight,activation,delta,value,changeInWeights):
        super().__init__(weight,activation,delta)
        ## corresponds to value of input node
        self.value = value
        # changeInWeights is set to 0 except for when implementing momentum
        self.changeInWeights = changeInWeights

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self,input_value):
        self._value = input_value

    @property
    def changeInWeights(self):
        return self._changeInWeights

    @changeInWeights.setter
    def changeInWeights(self,weightChange):
        self._changeInWeights = weightChange


class hiddenNode(Node):
    def __init__(self,weight,bias,activation,delta,changeInWeights):
        super().__init__(weight,activation,delta)
        self.bias = bias
        self.changeInWeights = changeInWeights
    
    @property
    def bias(self):
        return self._bias
    @bias.setter
    def bias(self,bias):
        self._bias = bias

    @property
    def changeInWeights(self):
        return self._changeInWeights
    @changeInWeights.setter
    def changeInWeights(self,weightChange):
        self._changeInWeights = weightChange
    

class outputNode(Node):
    def __init__(self,weight,bias,activation,delta,value):
        super().__init__(weight,activation,delta)
        self.bias = bias
        self.value = value

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self,input_value):
        self._value = input_value

    @property
    def bias(self):
        return self._bias
    @bias.setter
    def bias(self,bias):
        self._bias = bias




