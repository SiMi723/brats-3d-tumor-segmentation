import numpy as np

from data_preprocessing.label_mapping import (
    remap_labels,
    restore_labels,
)


original = np.array([
    [0, 1, 2, 4],
    [4, 2, 1, 0],
])


mapped = remap_labels(original)

restored = restore_labels(mapped)


print("Original:")
print(original)

print("\nMapped:")
print(mapped)

print("\nRestored:")
print(restored)

print("\nMapping correct:")
print(np.array_equal(original, restored))