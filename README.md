
# AML Video
### Homework
1. Choose two of the strategies presented, explain how they work.
1. Explain why you consider that these strategies would improve the performance of the baseline TSN (do not confuse TSN -code provided here- with TSM -last paper presented-).
1. Implement the strategies you chose and discuss your results.

Tip: It is probably easier to implement and analyse each strategy independently. 

**Bonus:** Implement simultaneously the non-local block or the TSM module in the forward function and discuss your results

### Mini UCF-101 Dataset (20 Classes)
Just download and untar:

[Train Data](https://filedn.com/l0kNCNuXuEq70c3iUHsXxJ7/train.tar) (2.6 GB)

[Val Data](https://filedn.com/l0kNCNuXuEq70c3iUHsXxJ7/val.tar) (1.1 GB)
 
### Running TSN Demo
Make sure you fix the path to the dataset train and val directories, then run:

```
python TSN.py
```

Check slides and in code comments for some extra info on the code.
