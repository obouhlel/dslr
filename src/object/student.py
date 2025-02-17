class Student:
    def __init__(self, index, house, first_name, last_name, birthday, best_hand,
                 arithmancy, astronomy, herbology, defense, divination, muggle_studies,
                 ancient_runes, history_of_magic, transfiguration, potions,
                 care_of_magical_creatures, charms, flying):
        self.index = index
        self.house = house
        self.first_name = first_name
        self.last_name = last_name
        self.birthday = birthday
        self.best_hand = best_hand
        self.arithmancy = arithmancy
        self.astronomy = astronomy
        self.herbology = herbology
        self.defense = defense
        self.divination = divination
        self.muggle_studies = muggle_studies
        self.ancient_runes = ancient_runes
        self.history_of_magic = history_of_magic
        self.transfiguration = transfiguration
        self.potions = potions
        self.care_of_magical_creatures = care_of_magical_creatures
        self.charms = charms
        self.flying = flying

    def __repr__(self):
        return (f"Student(Index={self.index}, House={self.house}, "
                f"First Name={self.first_name}, Last Name={self.last_name})")