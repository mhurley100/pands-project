# Validate from petal width Iris species classification

class Rectangle:
  def __init__(self, x1, y1, x2, y2):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2

  def overlaps(self, other):
    if self.x1 < other.x2 and self.x2 > other.x1 and self.y2 > other.y1 and self.y1 < other.y2:
      return True
    else:
      return False

r1 = Rectangle(1, 3, 5, 6)
r2 = Rectangle(3, 2, 7, 4)

print(r1.x1, r1.y1, r1.x2, r1.y2)
print(r2.x1, r2.y1, r2.x2, r2.y2)

print(r1.overlaps(r2))
print(r2.overlaps(r1))