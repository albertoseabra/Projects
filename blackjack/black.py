import random


class Card:
    """
    will have a number and a suit
    """
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit


class Deck:
    """
    will have 52 cards, 13 for each suit with values, in order: 2,3,4,5,6,7,8,9,10,J,Q,K,A
    """
    suits = ['clubs', 'diamonds', 'hearts', 'spades']
    values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    def __init__(self):
        self.deck = []
        for suit in self.suits:
            for value in self.values:
                self.deck.append(Card(value, suit))

    def deal(self, player):
        # deals a card, updates player hand and removes the card from the deck
        card_dealt = random.choice(self.deck)
        player.hand.append(card_dealt)
        self.calculate_hand_value(player)
        self.deck.remove(card_dealt)

    @staticmethod
    def card_value(card):
        # returns the card value
        if card.value.isdigit():
            return int(card.value)
        elif card.value == 'A':
            return 11
        else:
            return 10

    def calculate_hand_value(self, player):
        player.hand_value = 0
        for card in player.hand:
            player.hand_value += self.card_value(card)

        # if the value of the hand is bigger than 21 and the player has an Ace on hand the value of the Ace is
        # just 1 instead of the usual 11, we do that by subtracting 10 to the value of the player hand
        for i, card in enumerate(player.hand):
            if ('A' in player.hand[i].value) and (player.hand_value > 21):
                player.hand_value -= 10


class Player:
    """
    will start with an empty hand and some chips, will play the game
    """
    def __init__(self, name, chips):
        self.name = name
        self.chips = chips
        self.hand = []
        self.pot = 0
        self.hand_value = 0

    def bet(self, amount):
        self.chips -= amount
        self.pot = 2*amount

    def display_hand(self):
        print(self.name, 'your cards are:')
        for card in self.hand:
            print(card.value, card.suit)
        print(self.name, 'your cards value is:', self.hand_value)


class PokerGame:
    """
    the game initializes with a deck, the house and an empty list of players
    """
    def __init__(self):
        self.deck = Deck()
        self.house = Player('House', 99999999)
        self.players = []

    def add_player(self, name, chips):
        self.players.append(Player(name, chips))

    def first_round(self):
        # gives the first card to every player, including house
        self.deck.deal(self.house)
        for player in self.players:
            self.deck.deal(player)

    def players_turn(self, player):
        # does the player turn, if the player goes over 21 it will give him a hand value of 0
        print('Its your turn now', player.name)
        player.display_hand()
        while True:
            try:
                bet = int(input('How much do you want to bet?'))
            except ValueError:
                print('your bet needs to be an integer')
                continue
            if bet <= player.chips:
                player.bet(bet)
                break
            else:
                print("Who are you trying to fool? you don't have that many chips")
        self.deck.deal(player)
        player.display_hand()
        while True:
            if input('Do you want more cards ' + player.name + ' ?').lower() == 'y':
                self.deck.deal(player)
                player.display_hand()

                if player.hand_value > 21:
                    print('You went over 21, you lost!!!')
                    player.hand_value = 0
                    break
            else:
                print(player.name, 'final hand value is:', player.hand_value, "it's time for other player/house.")
                break

    def check_best_hand(self):
        # returns a tuple with the index of the player with the best hand and the value of that hand. Also prints
        best_hand = ()
        for index, player in enumerate(self.players):
            if not best_hand:
                best_hand = (index, player.hand_value)
            elif player.hand_value > best_hand[1]:
                best_hand = (index, player.hand_value)
            else:
                continue
        print('The best hand is:', best_hand[1], 'from player', self.players[best_hand[0]].name)
        return best_hand

    def house_turn(self, best_hand):
        # houses turn, tries to beat the best hand, if goes over 21 returns False
        print("It´s now House turn to play.")
        self.deck.deal(self.house)
        while self.house.hand_value < best_hand:
            self.deck.deal(self.house)
        self.house.display_hand()

        if self.house.hand_value > 21:
            return False
        else:
            return True

    def play_game(self):
        self.first_round()
        for player in self.players:
            self.players_turn(player)

        best_player, best_hand = self.check_best_hand()

        if best_hand == 0:
            print('Everyone went over 21, House wins!!')
        else:
            if self.house_turn(best_hand):
                print('House wins with a hand value of', self.house.hand_value, 'against',
                      self.players[best_player].name, 'with a hand value of', best_hand)
            else:
                print('house went over 21, player', self.players[best_player].name, 'wins with a hand value of', best_hand)
                self.players[best_player].chips += self.players[best_player].pot

    def display_chips(self):
        for player in self.players:
            print(player.name, 'has', player.chips, 'chips')

    def reset_deck_and_hands(self):
        # resets the deck of cards, the players hands and the players hands values to keep playing
        self.deck = Deck()
        self.house.hand = []
        self.house.hand_value = 0
        for player in self.players:
            player.hand = []
            player.hand_value = 0

    def adding_players(self):
        while True:
            try:
                num_players = int(input('How many human players will be playing the game?'))
                break
            except ValueError:
                print('The number of players needs to be an integer!')
        for player in range(1, num_players + 1):
            print('What is the name of player', player, '?')
            player_name = input('>>>')
            while True:
                try:
                    player_chips = int(input('With how many chips will ' + player_name + ' start?'))
                    break
                except ValueError:
                    print('The number of chips needs to be an integer!')
            self.add_player(player_name, player_chips)


def main():
    print('Welcome to the Poker Game, where you can try your luck against friends and/or the computer.')
    game = PokerGame()
    game.adding_players()

    while True:
        print('Its time to start!!')
        game.play_game()
        game.display_chips()
        if input('That was fun! Want to play again?').lower() == 'y':
            game.reset_deck_and_hands()
            continue
        else:
            print('Bye')
            break


if __name__ == "__main__":
    main()
