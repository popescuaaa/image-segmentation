# Image Segmentation using quadratic tree method
🌠

---
- Region based segmentation using a quad tree 
- Region growing for merging based on touching neighbours in all directions
- min-max homogeneity principle for image splitting

Note: is closer to KNN merging than region merging as I take in consideration the 
successsors and the neighbours which will result in a more diverse and more computation 
complex way of merging.

### Computation
The whole process is realized in parallel, using up to 100 workers which process and merge
separated parts of the list representation of the quad tree.


