"""
# 一、Python 程序设计：21 点（Blackjack）游戏

使用 Python 的 面向对象编程（OOP） 方式，实现一个可以在控制台中运行的简化版 21 点（Blackjack）游戏。

## 游戏规则简述（简化版 Blackjack）
1. 牌的点数

* **数字 2–10**：点数等于牌面数字
* **J、Q、K**：均算作 **10 点**
* **A（Ace）**：可以算作 **11 或 1**（自动转换）

2. 起始牌

玩家和电脑（庄家）各发两张牌。

3. 胜负判定（两张牌为 “Blackjack”）

* 一开始两张牌刚好为 **21 点**，则为 *Blackjack*，自动胜利
  （除非双方同时 Blackjack）

4. 爆牌（Bust）

* 超过 **21 点即为爆牌**，直接失败。

5. 玩家回合

玩家可以选择：

* **继续抽牌（Hit）**
* **停止抽牌（Stand）**

6. 电脑回合

* 庄家必须继续抽牌，直到点数达到 **17 或以上** 才停止
* 与玩家出现爆牌的情况比较胜负

7. 最终比较得分，判定胜负

## 实现要求
1. 玩家，电脑，游戏流程利用上面向对象编程
2. 要有测试功能的代码
"""
import random


# 牌堆
class card_base:
    def __init__(self):
        self.cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K', 'A']
        self.cards_num = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    # 单次摸牌
    def get_card(self):
        while True:
            random_num = random.randint(0, 12)
            if self.cards_num[random_num] > 0:
                self.cards_num[random_num] -= 1
                return self.cards[random_num]

    # 初始化双方手牌
    def get_first_card(self):
        result = list()
        for i in range(2):
            result.append([self.get_card() for _ in range(2)])
        return result


# 玩家类
class player:
    def __init__(self, name, cards):
        self.name = name
        self.cards = cards
        self.is_stand = False  # 是否继续摸牌


# 单局游戏类
class game:
    def __init__(self):
        self.card_base = card_base()
        cards = self.card_base.get_first_card()
        self.human = player('human', cards[0])
        self.computer = player('computer', cards[1])

    # 计算手牌点数
    @staticmethod
    def card_num_count(player):
        cards = player.cards
        count = 0
        count_A = 0
        for i in cards:
            if i in ['J', 'Q', 'K']:
                count += 10
            elif i == 'A':
                count_A += 1
                count += 11
            else:
                count += i
            if count > 21 and count_A:
                count_A -= 1
                count -= 10
        return count

    # 玩家摸牌
    def Hit(self, player):
        player.cards.append(self.card_base.get_card())

    # 玩家停牌
    def Stand(self, player):
        player.is_stand = True

    # 返回双方手牌
    def print_cards(self):
        return f"\n玩家手牌是{self.human.cards}\n电脑手牌是{self.computer.cards}\n"

    # 启动单次牌局
    def game_start(self):
        while True:
            human_num = self.card_num_count(self.human)
            computer_num = self.card_num_count(self.computer)
            # 判断是否满足21点
            if human_num == 21:
                print('玩家胜', self.print_cards())
                return 1
            elif computer_num == 21:
                print('玩家失败', self.print_cards())
                return 0
            elif human_num > 21:
                print('玩家失败', self.print_cards())
                return 0
            elif computer_num > 21:
                print('玩家胜', self.print_cards())
                return 1
            else:
                # 停止摸牌后比较点数
                if self.human.is_stand and self.computer.is_stand:
                    print(
                        f"玩家胜" + self.print_cards() if human_num >= computer_num else f"玩家失败" + self.print_cards())
                    return 1
                print(self.print_cards())
                print(f"玩家做出选择")
                choice = input("请输入Hit或者Stand:")
                # 宽松匹配命令
                if choice.lower() in 'hit':
                    self.Hit(self.human)
                elif choice.lower() in 'stand':
                    self.Stand(self.human)
                if computer_num >= 17:
                    self.Stand(self.computer)
                else:
                    self.Hit(self.computer)


# 单次棋局测试
if __name__ == '__main__':
    print("欢迎来到21点游戏")
    game = game()
    game.game_start()
