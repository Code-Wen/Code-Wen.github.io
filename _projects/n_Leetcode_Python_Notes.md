---
title: "Leetcode Algorithm Questions in Python 3"
excerpt: "Algorithmic thinking is a crucial component in every profession that involves coding, including data science. Here I'm documenting my solutions to algorithm questions on Leetcode, in hope that it would be of help to anyone interested."
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

## 3. Longest Substring Without Repeating Characters (M)

**Analysis:** `s_n` is the longest substring without repeating characters which ends at the `n`th position. `L_n = len(s_n)` and `L` is the current maximal length.  

```
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        L = 0
        s_n = ''
        L_n = 0
        dic = set([])
        if len(s) == 0:
            return L
        else:
            for n in range(len(s)):
                if s[n] in dic:
                    s_n = s_n[s_n.index(s[n])+1: ]+s[n]
                    dic = set(s_n)
                    L_n = len(dic)
                else:
                    s_n = s_n + s[n]
                    dic.add(s[n])
                    L_n += 1
                    if L_n > L:
                        L = L_n
            return L
```
## 4. Median of Two Sorted Arrays (H)

For a detailed explanation of the solution, see https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2481/Share-my-O(log(min(mn))-solution-with-explanation.
```
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m, n = len(nums1), len(nums2)
        if m > n:
            nums1, nums2, m, n = nums2, nums1, n, m
        if n == 0:
            raise ValueError

        imin, imax, half_len = 0, m, (m + n + 1) // 2
        while imin <= imax:
            i = (imin + imax) // 2
            j = half_len - i
            if i < m and nums2[j-1] > nums1[i]:
                # i is too small, must increase it
                imin = i + 1
            elif i > 0 and nums1[i-1] > nums2[j]:
                # i is too big, must decrease it
                imax = i - 1
            else:
                # i is perfect

                if i == 0: 
                    max_of_left = nums2[j-1]
                elif j == 0: 
                    max_of_left = nums1[i-1]
                else: 
                    max_of_left = max(nums1[i-1], nums2[j-1])

                if (m + n) % 2 == 1:
                    return max_of_left

                if i == m: 
                    min_of_right = nums2[j]
                elif j == n:
                    min_of_right = nums1[i]
                else:
                    min_of_right = min(nums1[i], nums2[j])

                return float((max_of_left + min_of_right) / 2)

```
## 21. Merge Two Sorted Lists

[Link](https://leetcode.com/problems/merge-two-sorted-lists/)

```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        s = ListNode(0)
        current = s
        
        if not l1:
            return l2
        if not l2:
            return l1
        else:
            
            while l1 and l2:
                if l1.val < l2.val:
                    current.next = l1
                    current = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    current = l2
                    l2 = l2.next
        
        
            if l1:
                current.next = l1
            elif l2: 
                current.next = l2
        
        return s.next
            
```
## 83. Remove Duplicates from Sorted List

[Link](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)


```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        current = head
        
        if head:
            next_node = head.next
            while next_node:
                if current.val != next_node.val:
                    current.next = next_node
                    current = next_node
                    next_node = current.next
                else:
                    next_node = next_node.next
            current.next = None
            return head 
        else:
            return head
```
## 876. Middle of the Linked List

[Link](https://leetcode.com/problems/middle-of-the-linked-list/)


```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        nxt = head.next
        current = head
        L=1
        while nxt:
            nxt = nxt.next
            L +=1
        for i in range(L//2):
            current = current.next
        return current
```

## String Permutation

**Problem Statement:** Given a string, write a function that uses recursion to output a list of all the possible permutations of that string.  If a character is repeated, treat each occurence as distinct, for example an input of `'xxx'` would return a list with 6 "versions" of `'xxx'`

For example, given `s='abc'` the function should return `['abc', 'acb', 'bac', 'bca', 'cab', 'cba']`

```
def perm(s):
    l = []
    if len(s) <= 1:
        l.append(s)
    else:
        for i in range(len(s)):
            for w in perm(s[1:]):
                l.append(s[0]+w)
            s = s[1:]+s[0]
    return l
```

## 141. Linked List Cycle

[Link](https://leetcode.com/problems/linked-list-cycle/)

The first solution is simply recording all the appeared nodes in a set, and traverse the linked list. Each step we check whether the next node is in the appeared list, if yes, then there is cycle, otherwise no cycle.

```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        appeared =set([])
        while head:
            if head not in appeared:
                appeared.add(head)
                head = head.next
            else:
                return True
        return False
```
The second approach is to traverse the linked list in two difference paces, one is `slow` and one is `fast`. If there is a cycle, then before the fast one hit `None`, he will catch up with the slow one. 

```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None or head.next is None:
            return False
        else:
            slow = head
            fast = head.next
            while fast.next and fast.next.next:
                if slow == fast:
                    return True
                else:
                    slow = slow.next
                    fast = fast.next.next
            return False
```
## 2. Add Two Numbers

```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        current_value = l1.val +l2.val
        
        l3 = ListNode(current_value % 10)
        head = l3
        
        while l1.next and l2.next:
            
            l1 = l1.next
            l2 = l2.next
            
            
            if current_value > 9:
                
                next_value = l1.val + l2.val + 1
            else:
                next_value = l1.val + l2.val
            
            current_value = next_value
            l3.next = ListNode(current_value % 10)
            l3 = l3.next
        
        while l1.next:
            l1 = l1.next
            
            if current_value > 9:
                
                next_value = l1.val + 1
            else:
                next_value = l1.val 
            
            current_value = next_value
            l3.next = ListNode(current_value % 10)
            l3 = l3.next
        
        
        while l2.next:
            l2 = l2.next
            
            if current_value > 9:
                
                next_value = l2.val + 1
            else:
                next_value = l2.val 
            
            current_value = next_value
            l3.next = ListNode(current_value % 10)
            l3 = l3.next
        
        if current_value > 9:
            l3.next =ListNode(1)
            
        return head
        
```
## 125. Valid Palindrome

```
class Solution:
    def isPalindrome(self, s: str) -> bool:
        import re

        regex = re.compile('[^a-zA-Z0-9]')
        #First parameter is the replacement, second parameter is your input string
        s = regex.sub('', s)
        
        
        
        L= len(s)
        i=0

        if len(s) <= 1:
            return True
        else:
            while i <= len(s)//2 - 1:
                
                if s[i].lower() == s[-(i+1)].lower():
                    i += 1
                else:
                    return False
            return True
```
## 680. Valid Palindrome II

```
class Solution:
    def check(self, s:str) -> bool:
        if len(s) <=1:
            return True
        else:
            i=0
            while i <= len(s)//2-1:
                if s[i] != s[-i-1]:
                    return False
                else:
                    i +=1
            return True
           
            
        
    def validPalindrome(self, s: str) -> bool:
        if len(s)<=2:
            return True
        
        else:
            i=0
            while s[i] == s[-i-1]:
                if i == len(s)//2 -1:
                    return True
                else:
                    i+=1
                
            
            if self.check(s[i+1:len(s)-i]) != True:
                return self.check(s[i:len(s)-i-1])
            return True
```

The same idea implemented slightly differently:
```
class Solution:
   def check(self, ss):
       # two pointers
       # move toward center if same
       
       l = 0
       r = len(ss) - 1
           
       while l < r:
           if ss[l] == ss[r]:
               l += 1
               r -= 1
           else:
               return (False, l, r)
       return (True, 0,0)
   
   def validPalindrome(self, s: str) -> bool:
       
       out, l, r = self.check(s)
       if out:
           return True
       # if not the same, check if s[l+1] == s[r] or s[l] == s[r+1], del flag
       
       else:
           return self.check(s[l+1:r+1])[0] or self.check(s[l: r])[0]
```
## 605. Can Place Flowers

Using recursion:

```
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        if n == 0:
            return True
        elif n > 0 and len(flowerbed)==0:
            return False
        else:
            if flowerbed[0]==1:
                if len(flowerbed) >2:
                    return self.canPlaceFlowers(flowerbed[2:], n)
                else:
                    return False
            if flowerbed[0] == 0:
                if len(flowerbed) ==1:
                    n -= 1
                    flowerbed[0] =1
                    return self.canPlaceFlowers(flowerbed, n)
                if len(flowerbed) >1:
                    if flowerbed[1] ==0:
                        flowerbed[0] = 1
                        n -= 1
                        return self.canPlaceFlowers(flowerbed, n)
                    else:
                        if len(flowerbed) > 3:
                            return self.canPlaceFlowers(flowerbed[3:],n)
                        else:
                            return False
            
```

Using iteration:

```
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        i = 0
        L= len(flowerbed)
        while i < L:
            if n == 0:
                return True
            
            if flowerbed[i] == 1:
                i+=2
            elif flowerbed[i] == 0 and i+1<L:
                if flowerbed[i+1] == 1:
                    i+=3
                else:
                    n-=1
                    i+=2
            else:
                i+=2
                n-=1
        return n<=0
```

## 292. Nim Game

We can actually get the general formula for the outcome: if n is divisible by 4, then the first player loses; otherwise he will win. 
```
class Solution:
    def canWinNim(self, n: int) -> bool:
        if n%4 ==0:
            return False
        else:
            return True
```
## 283. Move Zeroes
```
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i=0
        num_zeros=0
        j=0
        while i<len(nums):
            if nums[i]==0:
                i+=1
                num_zeros+=1
            else:
                if num_zeros == 0:  ## if no zeros appeared before, simply move forward
                    i+=1
                    j+=1
                else:            ## otherwise,interchange 0 and the non-0 number then forward
                    nums[j]=nums[i]
                    nums[i]=0
                    j+=1
                    i+=1
```
## 278. First Bad Version

Typical binary search
```
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        if isBadVersion(n) == False:
            return "No bad!"
        else:
            head =1
            tail =n
            mid=(head+tail)//2
            while mid>head:
                if isBadVersion(mid):
                    tail= mid
                    mid=(head+tail)//2
                else:
                    head = mid
                    mid=(head+tail)//2
                    
            if isBadVersion(mid):
                return head
            else:
                return tail
```
## 374. Guess Number Higher or Lower

Simply binary search.

```
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num):

class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        head = 1
        tail = n
        mid =(head+tail)//2
        while mid>head:
            if guess(mid)==-1:
                tail=mid
                mid=(head+tail)//2
            elif  guess(mid)==1:
                head=mid
                mid=(head+tail)//2
            else:
                return mid
        if guess(head)==0:
            return head
        else:
            return tail
```
## 371. Sum of Two Integers

A nice exercise on bit manipulations. 
```
class Solution:
    def getSum(self, a: int, b: int) -> int:
        if a == -b:
            return 0
        if abs(a) > abs(b):
            a, b = b, a
        if a < 0:
            return -self.getSum(-a, -b)
        while b:
            c = a & b
            a ^= b
            b = c << 1
        return a
                
```

## 342. Power of Four

If `num` is non-positive, False.
Check if it is a power of 2: if `num & (num-1)` is 0 then we are good.
Check if it is a power of 4: if `num%3==1`, we are good.

```
class Solution:
    def isPowerOfFour(self, num: int) -> bool:
        if num <= 0:
            return False
      
        else:
            
            return  not(num & (num-1)) and num%3==1
```

## 155. Min Stack

In order to retrieve the minimal value in constant time, we sacrifice space to store the minimal values at each step of the stack.

```
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.val = []
        self.min = float('inf')
        self.min_record = []
        

    def push(self, x: int) -> None:
        self.val.append(x)
        if self.min > x:
            self.min = x
            self.min_record.append(x)
        elif self.min == x:
            self.min_record.append(x)
        else:
            self.min_record.append(self.min)

    def pop(self) -> None:
        if len(self.val)>0:
            
            self.val.pop()
            self.min_record.pop()
            if len(self.val)>0:
                self.min = self.min_record[-1]
            else:
                self.min = float('inf')
        else:
            print ('There is nothing to pop!')

    def top(self) -> int:
        return self.val[-1]

    def getMin(self) -> int:
        return self.min


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

```
## 345. Reverse Vowels of a String

```
class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = set(['a','e','i','o','u'])
        L = len(s)
        H = 0
        T = L-1
        temp = list(s)
        while H < T:
            while temp[H].lower() not in vowels and H < T:
                H += 1
            while temp[T].lower() not in vowels and T > H:
                T -= 1
            
            if H < L and T >= 0:
                temp[H], temp[T] = temp[T], temp[H]
                
                H += 1
                T -= 1
            
            
        
        return ''.join(temp)
```
## 443. String Compression
 
Leetcode is not working for my code. It works well on Spyder with Python 3.7 on my laptop.

```
class Solution:
    def compress(self, chars: List[str]) -> int:
        
        L=len(chars) 
        current=1           ## track the location of the modified list
        step=1              ## step is to keep track of where we are at in the original list
        count=1             ## track the count of continuous appearance of the letter
        
        ## if length is no larger than 1, no need to modify
        
        if L<=1:
            return L
            
        else:
            letter=chars[0]
            while step < L:
                ## if the next letter is the same as the current letter, count+1, step+1 and pop it out
                
                if letter == chars[current]:
                    count += 1
                    step += 1
                    chars.pop(current)
                
                ## if the next letter is different with the current letter, we update the list
                
                else:
                    ## if count is 1, we do not need to add a number, so just move on to the next letter
                    
                    if count == 1:
                        
                        letter = chars[current]
                        current += 1
                        step += 1
                    ## if count>1, we need to add a number after the current letter, then move on the next letter
                    
                    else:
                        letter = chars[current]
                        temp = list(str(count))
                        chars = chars[:current]+temp+chars[current:]
                        current += len(temp)+1
                        step += 1
                        count = 1
                        
            ## edge case: after getting to the end of the list, we need to attach the count number if needed
            
            if count > 1:
                temp = list(str(count))
                chars = chars+temp
                print(chars)
            
            return len(chars)
                        
```
## 367. Valid Perfect Square

```
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        L=len(str(num))//2
        temp=0
        while L>=0:
            i=9
            
            while (temp+i*(10**L))**2 > num:
                i-=1
            
            temp=temp+i*(10**L)
            L-=1
            
        return (temp**2) == num
        
```
## 414. Third Maximum Number
```
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        max1=nums[0]
        max2=float('-inf')
        max3=float('-inf')
        i=0
        for i in range(len(nums)):
            if nums[i] > max1:
                max3, max2 = max2, max1
                max1 = nums[i]
                i+=1
            elif nums[i] == max1 or nums[i]==max2 or nums[i]==max3:
                i+=1
            elif nums[i] < max1 and nums[i] > max2:
                max2, max3 = nums[i], max2
                i+=1
            elif nums[i]< max2 and nums[i] > max3:
                max3=nums[i]
                i+=1
                
            else:
                i+=1
        if len(set([max1,max2,max3]))==3 and max3>float('-inf'):
            return max3
        else:
            return max1
```
## 504. Base 7

```

class Solution:
    def convertToBase7(self, num: int) -> str:
        i=8
        sign = 1
        s=0
        if num==0:
            return '0'
        if num<0:
            sign = -1
            num = -num
            print(num)
        while i >= 0:
            power = 7**i
            j=1
            while j*power<=num:
                j+=1
            
            j-=1
            num = num-j*power
            s+=j*(10**i)
            i-=1
        
        return str(sign*s)
        
```
## 383. Ransom Note
```
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        
        
        
        dict1=dict(zip([chr(i) for i in range(ord('a'),ord('z')+1)], [0]*26))
        
        L1=len(ransomNote)
        L2=len(magazine)
        if L1==0:
            return True
        else:
            if L2==0:
                return False
            else:
                i=1
                while i<=L1:
                    dict1[ransomNote[i-1]] += 1
                    i+=1
                
                j=1
                while j<=L2:
                    dict1[magazine[j-1]] -=1
                    j+=1
                
            if max(dict1.values()) > 0:
                return False
            else:
                return True
```
## 541. Reverse String II
```
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        i=0
        L=len(s)//(2*k)
        remainder=len(s)%(2*k)
        while i < L:
            
            
            s = s[:2*k*i]+s[2*k*i:2*k*i+k][::-1]+s[2*k*i+k:]
            i+=1
            
        if remainder > k:
            s = s[:2*k*i]+s[2*k*i:2*k*i+k][::-1]+s[2*k*i+k:]
        elif remainder > 0:
           
            
            s = s[:2*k*i]+s[2*k*i:2*k*i+k][::-1]
        
            
        
        return s
            
```
## 557. Reverse Words in a String III
```
class Solution:
    def reverseWords(self, s: str) -> str:
        start=0
        end=0
        L=len(s)
        if L==0:
            return s
        else:
            while end < L:
                if s[end]!=' ':
                    end+=1
                else:
                    s=s[:start]+s[start:end][::-1]+s[end:]
                    start = end+1
                    end+=1
                    
            if start < L:
                s = s[:start]+s[start:][::-1]
```

## 682. Baseball Game

```
class Solution:
    def calPoints(self, ops: List[str]) -> int:
        clean =[]
        L=len(ops)
        i=0
        while i < L:
            if ops[i]=='C':
                clean.pop()
                i+=1
            elif ops[i]=="+":
                clean.append(clean[-2]+clean[-1])
                i+=1
            elif ops[i]=='D':
                clean.append(clean[-1]*2)
                i+=1
            else:
                clean.append(int(ops[i]))
                i+=1
        return sum(clean)
```
## 633. Sum of Square Numbers
```
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        j=0
        
        while j**2 <= c/2:
            if math.sqrt(c-j**2).is_integer():
                return True
            else:
                j+=1
        return False
        
```
## 520. Detect Capital

```
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        s=set([chr(i) for i in range(ord('A'),ord('Z')+1)])
        i=0
        cap_num=0
        if word[0] in s:
            for i in range(0,len(word)):
                if word[i] in s:
                    cap_num+=1
                    i+=1
                    
                else:
                    i+=1
            if cap_num==len(word) or cap_num==1:
                return True
            else:
                return False
            
        else:
            for i in range(0,len(word)):
                if word[i]  in s:
                    return False
                else:
                    i+=1
            return True
```
## 686. Repeated String Match

```
class Solution:
    def repeatedStringMatch(self, A: str, B: str) -> int:
        
        if len(set(B)-set(A))>=1:
            return -1
        else:
            
            LB=len(B)
            A_temp =A
            count=1
            while len(A_temp)<LB:
                A_temp = A_temp+A
                count+=1
            if B in A_temp:
                return count
            else:
                A_temp += A
                count+=1
                if B in A_temp:
                    return count
                else:
                    return -1
```
## 412. Fizz Buzz

```
class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        l=[str(i+1) for i in range(n)]
        i1=3
        i2=5
        i3=15
        while i1 < n+1:
            l[i1-1]="Fizz"
            i1 += 3
        while i2 < n+1:
            l[i2-1]='Buzz'
            i2 += 5
        while i3 < n+1:
            l[i3-1]='FizzBuzz'
            i3 += 15
        return l
```
## 977. Squares of a Sorted Array

__Main idea:__ Use two pointers to get the next largest square.
```
class Solution:
    def sortedSquares(self, A: List[int]) -> List[int]:
        answer = [0] * len(A)
        l, r = 0, len(A) - 1
        while l <= r:
            left, right = abs(A[l]), abs(A[r])
            if left > right:
                answer[r - l] = left * left
                l += 1
            else:
                answer[r - l] = right * right
                r -= 1
        return answer
```
## 970. Powerful Integers

First approach is brute-force. Be aware of the edge case that either x or y is 1 and the edge case that bound is less than 2.

```
class Solution:
    def powerfulIntegers(self, x: int, y: int, bound: int) -> List[int]:
        s=set()
        i,j=0,0
        sum=2
        x,y=max(x,y),min(x,y)
        
        if x==1:
            if bound >=2:
                return [2]
            else:
                return []
        elif y==1:
            while sum<=bound:
                s.add(sum)
                i+=1
                sum=x**i + 1
        else:
            
            while sum<=bound:
                s.add(sum)
                j+=1
                sum=1+y**j


            while j>=0:
                i=1
                sum=x**i+y**j
                while sum<=bound:
                    s.add(sum)
                    sum=x**i+y**j
                    i+=1
                j-=1

        return list(s)
```

Second approach is to record the powers of x up to bound, and the powers of y up to bound. Then add the two lists while keeping the sums which is less than or equal to the bound.

```
class Solution:
    def powerfulIntegers(self, x: int, y: int, bound: int) -> List[int]:
        if bound < 2:
            return []
        
        
        xpower,ypower,i,j=[],[],0,0
        if x==1:
            xpower=[1]
        else:
            while x**i<=bound:
                xpower.append(x**i)
                i+=1
        
        if y==1:
            ypower=[1]
        else:
            while y**j<=bound:
                ypower.append(y**j)
                j+=1
                
        return list(set([x+y for x in xpower for y in ypower if x+y<=bound]))
        
```
## 290. Word Pattern

Idea: use a dictionary to keep track of the pattern. Be aware of that the unique letters in `pattern` and the unique words in the string should be the same.

```
class Solution:
    def wordPattern(self, pattern: str, str: str) -> bool:
        s=str.split()
        if len(pattern)!=len(s) or len(set(pattern))!=len(set(s)):
            return False
        else:
            tracking = {}
            i=0
            while i<len(pattern):
                if pattern[i] not in tracking.keys():
                    tracking[pattern[i]]=s[i]
                    i+=1
                else:
                    if tracking[pattern[i]] != s[i]:
                        return False
                    else:
                        i+=1
            return True
```
## 645. Set Mismatch
```
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        n=len(nums)
        missing = list(set([i+1 for i in range(n)])-set(nums))[0]
        duplicate =  int(-n*(n+1)/2 +sum(nums)+missing)
        return [ duplicate,missing]
```

## 287. Find the Duplicate Number

This problem can be solved using the [Floyd's Tortoise and Hare](https://en.wikipedia.org/wiki/Cycle_detection) or Floyd's cycle detection algorithm. Essentially we are treating the list with duplicates as a linked list with cycles.

```
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow=nums[0]
        fast=nums[nums[0]]
        
        ## identifying whether there is cycle or not
        
        while slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]
        
        ## now find the entry of the cycle, where our duplicate resides. 
        
        fast=0
        while slow != fast:
            fast = nums[fast]
            slow = nums[slow]
        return slow
```