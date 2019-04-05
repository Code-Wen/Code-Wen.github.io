---
title: "Leetcode Algorithm Questions in Python"
excerpt: "A good way of learning is through practice. This is my log of practicing Leetcode problems in Python."
collection: projects
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
        
        seq = "1"
        if n >= 2:
            for k in range(0,n-1):
                list = [i for i in seq]
                stack = []
                while len(list) > 0:
                    a = list[0]
                    l = 0
                    while len(list) > 0 and a == list[0]:
                        l += 1
                        list.remove(a)
                    stack.append(str(l)+a)
                    
                seq = ''.join(stack)
        return seq
```

## 70. Climbing Stairs

[Link to the problem](https://leetcode.com/problems/climbing-stairs/)

**Analysis:**

1. This is again a recursive process.   
2. In order to climb *n+1* stairs, one has exactly two possibilities for the previous step: either one reach the *n*th stair, or one reach the *(n-1)*th stair. Thus if we denote by `a(k)` the number of different ways to climb *n* stairs, then we have `a(n+1)=a(n)+a(n-1)` (here we need to assume `n-1>0`). This is the inductive formula!
3. Set up the initial terms `a(0)=a(1)=1` and use the above inductive formula, we can easily get it done via a simple `for` loop.

```
class Solution:
    def climbStairs(self, n: int) -> int:
        a = [1,1]
        for k in range(0,n):
            a.append(a[k]+a[k+1])
            
        return a[n]
```

**Remark:** If one realizes that this is actually the Fibonacci sequence, he could choose to use the explict formula as in [Wikipedia: Fibonacci number](https://en.wikipedia.org/wiki/Fibonacci_number).

## 344. Reverse String

[Link to the problem](https://leetcode.com/problems/reverse-string/)

**Analysis:** Use `head` to store the head of the unchanged part of the list, and use `tail` to store the tail. Then exchange their positions.

```
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
       
        L = (len(s)+1)//2
        for k in range(0, L):
            head = s[k]
            tail = s[-k-1]
            s[k] = tail
            s[-k-1] = head

```

## 58. Length of Last Word

[Link to the problem](https://leetcode.com/problems/length-of-last-word/)

**Analysis:** First use the `.strip()` method to remove any empty space in the beginning and in the end of the string. Then count how many non-empty letters in the tail (from right to left) until we reach to the first empty space.

```
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        k = 0
        stripped = s.strip()       
        while k <= len(stripped)-1:
            if stripped[-k-1] != " ":
                k += 1
            else:
                return k
        
        return k
```

## 231. Power of Two

[Link to the problem](https://leetcode.com/problems/power-of-two/)

**Analysis:** One can of course do it in a recursive fashion, that is, divide `n` by 2, check the remainder: if it is non-zero, stop and return `False`; otherwise, update `n=n/2` and repeat. If in the end we get `n=1`, return `True`.

However there is an clever one-liner using the bitwise operations such as `&` and `|`:

```
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n>0 and n&(n-1) == 0
```

For a good explaination of the Python bitwise operators, see [Here](https://www.tutorialspoint.com/python/bitwise_operators_example.htm). So the idea of the above solution is: if `n<=0`, return `False`. For the second part, `n&(n-1)==0`, any positive integer `n` is represented uniquely as a sequence of bits, that is, `1`s and `0`s. However, `n` is a power of 2 if and only if the bit form of `n` looks like `10...0`, where there is only a leading `1`, and the rest are all 0s.  In that case, the bit form of `n-1` is `01..1`. Therefore the bitwise *and* operator `n&(n-1)` will give a sequence of 0s, since at each corresponding position for the bit forms of `n` and `n-1`, there is always a `0` and a `1`. Meanwhile, if `n` is not a power of 2, then in the bit form of `n`, there are at least two `1`s, say `...1...1...`. A moment thinking will lead to the conclusion that the bit form of `n-1` will have `1` in the first position, as `...1...*...`, therefore the outcome of `n&(n-1)` is never 0!


## 217. Contains Duplicate

[Link to the problem](https://leetcode.com/problems/contains-duplicate/)

**Analysis:** Change the list to set, then compare the lengths of the list and the set.

```
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        
        return len(set(nums)) != len(nums)
``` 

## 204. Count Primes

[Link to the problem](https://leetcode.com/problems/count-primes/)

**Analysis:**  

1. Build a logic list `res` to store whether the integers for 0 to n-1 are prime or not.
2. Initial conditions: 0 and 1 are not primes, so set the first two elements in `list` to be `False`, while set the third element (which corresponds to 2) to be `True`.
3. Using a `for` loop, we update the list `res`: if `i` is a prime, then `2*i`, `3*i`, etc. are all non-prime.
4. We count how many `True` are in `res`. 

```
class Solution:
    def countPrimes(self, n: int) -> int:
            if n <= 2:
                return 0
            res = [True] * n
            res[0] = res[1] = False
            for i in range(2, n):
                if res[i] == True:
                    for j in range(2, (n-1)//i+1):
                        res[i*j] = False
            return sum(res)
```

## 263. Ugly Number
[Link to the problem](https://leetcode.com/problems/ugly-number/)

**Analysis:**

1. If `num` is strictly greater than 1, we keep trying to reduce it, by dividing `num` by `2,3,5`, if divisible.
2. We use the logic variable `ugly` to track whether `num` is divisible by `2,3,5` or not.

```
class Solution:
    def isUgly(self, num: int) -> bool:
        while num > 1:
            ugly = False
            for k in [2,3,5]:
                if num % k == 0:
                    ugly = True
                    num = num/k
            if not ugly:
                return ugly
        return num == 1
```


## 202. Happy Number

[Link to the problem](https://leetcode.com/problems/happy-number/)

**Analysis:** 

1. Use a list called `history` to store the history of numbers we obtain. The variable `tail` is the current last element in `history`.

2. If `tail` is not 1, we continue the process to add numbers to `history`.

3. Each time we add an element to `history`, we check whether this element has appeared before. If so, then we will get an infinite loop, so we stop the loop.

4. Once we break the loop, we check whether `tail` is 1 or not. If 1, then this number is happy; otherwise, it is an unhappy number.

```
class Solution:
    def isHappy(self, n: int) -> bool:
        history = [n]
        tail = n
        while tail != 1:
            l = [int(i)**2 for i in str(tail)]
            tail = sum(l)
            history.append(tail)
            if len(history) > len(set(history)):
                break
        
        return tail == 1
```    

## 268. Missing Number

[Link to the problem](https://leetcode.com/problems/missing-number/)

**Analysis:** Sort the list first, then compare the position `k` with its value `nums[k]`.

```
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        nums.sort()
        for k in range(0,len(nums)):
            if nums[k] == k:
                k += 1
            else:
                break
        return k
```

Another direct solution (kinda cheating imo):

```
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        N = len(nums)
        return N(N+1)/2 - sum(nums)
```

## 171. Excel Sheet Column Number

[Link to the problem](https://leetcode.com/problems/excel-sheet-column-number/)

**Analysis:** First build a dictionary to indicate which number each letter in the alphabet to stand for. Then `s` in basically a number in the 26 system. We just need to transform it into a number in the decimal system.

```
class Solution:
    def titleToNumber(self, s: str) -> int:
        alphabet = [chr(i) for i in range(65, 91)]
        nums = [i for i in range(1, 27)]
        dic = dict(zip(alphabet, nums))
        L = len(s)
        s_list = [dic[s[i]] * (26 ** (L-1-i)) for i in range(0,L)]
     
        return sum(s_list)
```

## 169. Majority Element

[Link to the problem](https://leetcode.com/problems/majority-element/)

**Analysis:** 

1. Compute `L = len(nums)//2`, which is the least number of repetitions for the majority element.
2. Sort the list `nums`.
3. For each element `i` in the set formed from elements in `nums`, get its index in the sorted list `ind`, then check if the `ind + L`th element is equal to `i`. If so, we captured the majority element!
```
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        L = len(nums)//2
        nums.sort()
        for i in set(nums):
            ind = nums.index(i)
            if ind + L <= len(nums)-1: 
                if nums[ind + L] == i:
                    return i
```

## 121. Best Time to Buy and Sell Stock

[Link to the problem](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

**Analysis:** 

1. We use `MAX_step` to denote the maximal profit if one chooses to sell on the `k`th day. Note that this quantity can be negative.
2. `MAX_step` at `k` can be obtained from the `MAX_step` at `k-1`: it equals the maximal value between `prices[k]-prices[k-1]`  and `MAX_step+prices[k]-prices[k-1])`.
3. In the `for` loop, we inductively update the current max profit if one chooses to sell the stock on the `1<=i<=k` day.


```
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        MAX=0
        MAX_step=0
        L=len(prices)
        if L>=2:
            for k in range(1, L):
                MAX_step = max(prices[k]-prices[k-1], MAX_step+prices[k]-prices[k-1]) 
                MAX=max(MAX, MAX_step)
        
        return MAX
```
## 122. Best Time to Buy and Sell Stock II

[Link to the problem](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

**Analysis:** Now we allow multiply transactions, this will lead to different analysis.

1. We use `P` to denote the total profit, `buy` the last buy date, and `sell` the last sell date.
2. If the next day price is lower or equal to the price of the previous day, the `buy` date will move forward, to generate more profit. `buy` will stop move forward once positive profit is possible, that is, `prices[buy+1] > prices[buy]`.
3. Once `buy` is fixed, we consider `sell`. First set `sell = buy + 1` (recall that we will always have prices[sell] > prices[buy], due to how we work with `buy` in the previous step). `sell` will move forward if the price keeps increasing strictly. Once the price stops to increase strictly, `sell` stops moving forward.
4. Once we obtain `sell` and `buy` for the next transaction, we update the profit by `P=P+prices[sell]-prices[buy]`.

```
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
            P = 0  #profit
            buy = 0  #buy date
            sell = 0  #sell date
            L = len(prices)
            while buy < L-1:
                
                if prices[buy+1] <= prices[buy]:
                    buy += 1
                else:
                    sell = buy + 1
                    while sell < L-1 and prices[sell+1] > prices[sell]:
                        sell += 1
                    P = P + prices[sell]-prices[buy]
                    buy = sell + 1
            
            return P
```
## 198. House Robber

[Link to the problem](https://leetcode.com/problems/house-robber/)
```
class Solution:
    def rob(self, nums: List[int]) -> int:
        last, now = 0, 0
        
        for i in nums: last, now = now, max(last + i, now)
                
        return now
```
## 219. Contains Duplicate II

[Link to the problem](https://leetcode.com/problems/contains-duplicate-ii/)

**Analysis:** 

1. The idea is very similar to Problem 217. We just need to check that whether any chunk of length `k+1` contains duplicates.
2. Instead of use `set()` to convert a chunk of the list into a set, we use a little trick here: first we find the set from the first chunk. Then as we move to the right, we remove the first element `nums[step-1]` from the set, and then add the next element `nums[step+k]`  to the set.

```
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        L = len(nums)
        if L <= 1 or k == 0:
            return False
        elif L <= k+1:
            return len(set(nums)) != L
        else:
            s = set(nums[0:k+1])
            if len(s) != k+1:
                return True
            for step in range(1, L-k):
                s.remove(nums[step-1])
                s.add(nums[step+k])
                if len(s) != k+1:
                    return True
           
                    
            return False
```

## 172. Factorial Trailing Zeroes

**Analysis:** It is the same as count the exponent of 5 in the prime factorization of n-factorial. We first see how many numbers from 1 to n are divisible by 5 : `n//5`. For the numbers which can be divided by 25, we have to count them twice, the first time is already included in the previous step, so we need to add how many numbers can be divided by 25, which can be obtained by `(n//5)//5`. We continue until the remainder is 0.
 
```
class Solution:
    def trailingZeroes(self, n: int) -> int:
        if n == 0:
            return 0
        else:
            k=0
            while n > 0:
                k += n//5
                n = n//5
            return k
```
## 168. Excel Sheet Column Title

[Link to the problem](https://leetcode.com/problems/excel-sheet-column-title/)

```
class Solution:
    def convertToTitle(self, n: int) -> str:
        a = [chr(i) for i in range(65,91)]
        b = [i for i in range(0, 26)]
        order = dict(zip(b,a))
        s = ''
        while n > 0:
            s = order[(n-1)%26] + s
            n = (n-1)//26
        return s
```
## 167. Two Sum II - Input array is sorted

[Link to the problem](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

```
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        s = set([])
        for i in range(len(numbers)):
            if target-numbers[i] in s:
                return [numbers.index(target-numbers[i])+1,i+1]
            else:
                s.add(numbers[i])
```
