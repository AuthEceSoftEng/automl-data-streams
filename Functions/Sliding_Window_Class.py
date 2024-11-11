
class SlidingWindow:
    """
        class of sliding window used for evaluation

        functions:
            __init__(self, limit): Constructor that create a list with an upper limit
            add(self, item): Adds an item in the window
            get(self): Returns the list of the window

    """
    def __init__(self, limit):
        self.limit = limit
        self.list = []

    def add(self, item):
        if len(self.list) >= self.limit:
            self.list.pop(0)  # Remove the first item
        self.list.append(item)

    def get(self):
        return self.list

    def get_specific(self, index):
        return self.list[index]
