from pygraphblas import FP32

class ReLUNeuron(FP32):
    @binary_op(FP32)
    def TIMES(x, y):
        result = min(x + y, 32)
        if result < 0:
            return 0
        return result

ReLUNeuron_monoid = ReLUNeuron.new_monoid(FP32.MAX, ReLUNeuron.one)
ReLUNeuron_semiring = ReLUNeuron.new_semiring(ReLUNeuron_monoid, ReLUNeuron.TIMES)


