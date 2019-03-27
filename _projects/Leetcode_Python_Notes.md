---
title: "Leetcode Algorithm Questions in Python"
excerpt: "A good way of learning is through practice. This is my daily log of practicing Leetcode problems in Python."
collection: notes
---

## 38. Count and Say

[Link to the problem](https://leetcode.com/problems/count-and-say/)

**Analysis:**  
1. This is an inductive process. The *(k+1)*th reading depends on the *k*th reading.  
2. The starting point of the induction is the first reading, which we set as `seq="1"`.  
3. We need to recursively update `seq` *n-1* times.  
4. For each update, we use `list` to record the list of letters in the previoius `seq`. Starting with the first element `a=list[0]`, we count how many times `a` appears in the head of `list`, and record the number by `l`. In the process we also remove `a` in the head of `list`. Then we record the str `str(l)+a` in the list `stack`. Continue until we exhaust `list`, that is, when the length of `list` becomes 0.  In the last step we concatenate the strings in `stack` to form the new `seq`.   
5. Return `seq` which is the *n*th reading.

```
class Solution:
    def countAndSay(self, n: int) -> str:
        
        seq="1"
        if n>=2:
            for k in range(0,n-1):
                list=[i for i in seq]
                stack=[]
                while len(list)>0:
                    a=list[0]
                    l=0
                    while len(list)>0 and a==list[0]:
                        l+=1
                        list.remove(a)
                    stack.append(str(l)+a)
                    
                seq=''.join(stack)
        return seq
```

