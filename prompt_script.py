import random
# import argparse 

# parser = argparse.ArgumentParser(
#                     prog='prompt_script',
#                     description='generate prompts for generating training images',
#                     epilog='END')

# parser.add_argument('-N', '--num_prompts', type=int)        
# parser.add_argument('-a', '--num_attire', type=int)
# parser.add_argument('-s', '--subject', type=str)        
# parser.add_argument('-S', '--simple', nargs='?', type=bool)        

# args = parser.parse_args()


attire_categories = [
    "t-shirt",
    "shirt",
    "polo shirt",
    "tank top",
    "blouse",
    "sweater",
    "cardigan",
    "hoodie",
    "jacket",
    "blazer",
    "coat",
    "vest",
    "dress",
    "gown",
    "skirt",
    "jeans",
    "pants",
    "shorts",
    "leggings",
    "jumpsuit",
    "romper",
    "suit",
    "tuxedo",
    "saree",
    "kimono",
    "robe",
    "swimsuit",
    "bikini",
    "boardshorts",
    "sweatshirt",
    "tracksuit",
    "sportswear",
    "uniform",
    "costume",
    "sleepwear",
    "underwear",
    "socks",
    "shoes",
    "sneakers",
    "boots",
    "sandals",
    "heels",
    "flats",
    "loafers",
    "slippers",
    "flip-flops",
    "hat",
    "cap",
    "beanie",
    "headband",
    "scarf",
    "gloves",
    "belt",
    "tie",
    "bowtie",
    "necklace",
    "bracelet",
    "ring",
    "earrings",
    "sunglasses",
    "watch",
    "handbag",
    "backpack",
]


colors_complex = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "gray",
    "black",
    "white",
    "beige",
    "cream",
    "navy blue",
    "turquoise",
    "teal",
    "mint green",
    "lime green",
    "olive green",
    "forest green",
    "maroon",
    "burgundy",
    "coral",
    "salmon",
    "lavender",
    "violet",
    "indigo",
    "magenta",
    "rose",
    "gold",
    "silver",
    "bronze",
    "copper",
    "charcoal",
    "ivory",
    "khaki",
    "slate",
    "tan",
    "tawny",
    "ruby",
    "sapphire",
    "emerald",
    "amethyst",
    "topaz",
    "opal",
    "pearl",
    "onyx",
    "ruby red",
    "sapphire blue",
    "emerald green",
    "amber",
    "crimson",
    "midnight blue",
    "sunset orange",
    "rose gold"
]

colors_simple = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "indigo",
    "violet",
    "purple"
]


prompt_templates = [
        "a photo of a {subject} wearing {attire}"
]




# subject 
# prompts

def gen_prompts(num_prompts, num_attire, subject, simple=True):
    prompts = []
    if(simple):
        colors = colors_simple
    else:
        colors = colors_complex

    for i in range(num_prompts):
        prompt = random.choice(prompt_templates)
        attire = ""
        for j in range(num_attire):
            color = random.choice(colors)
            attire += color + " "
            item = random.choice(attire_categories)
            attire += item
            if(j >= 0):
                attire += ", "

        attire = attire.strip()
        prompt = prompt.format(subject=subject, attire=attire)
        # print(attire)
        # print(prompt)
        prompts.append(prompt)
    return prompts

    # for prompt in prompts:
    #     print(prompt)
