class RePlAcePlacer:
    def place(self, benchmark):
        # Simply return the positions coming from the benchmark loading (which are RePlAce)
        return benchmark.macro_positions

def get_placer():
    return RePlAcePlacer()
