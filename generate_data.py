import csv
import random



categories = {
    "Electronics": {
        "brands": ["Sony", "Samsung", "LG", "Apple", "Dell", "HP", "Bose", "Jabra", "Anker", "Lenovo", "Asus", "Acer"],
        "products": ["Smart TV", "Smartphone", "Laptop", "Wireless Headphones", "Bluetooth Speaker", "Tablet", "Smartwatch", "Power Bank", "Monitor", "Gaming Mouse", "Mechanical Keyboard", "Router"],
        "features": ["4K Ultra HD", "128GB Storage", "Active Noise Cancelling", "30-hour battery life", "Waterproof IP68", "16GB RAM, 512GB NVMe SSD", "120Hz OLED Display", "Fast Charging support", "Wi-Fi 6 Enabled", "Ultra-Slim Design"]
    },
    "Clothing": {
        "brands": ["Nike", "Adidas", "Puma", "Levi's", "Zara", "H&M", "Under Armour", "Tommy Hilfiger", "Calvin Klein", "Ralph Lauren", "Vans", "Converse"],
        "products": ["T-Shirt", "Jeans", "Sneakers", "Jacket", "Sweater", "Dress", "Shorts", "Hoodie", "Chinos", "Polo Shirt", "Running Shoes", "Track Pants"],
        "features": ["100% Organic Cotton", "Slim Fit", "Breathable Mesh", "Water Resistant", "Vintage Wash", "4-Way Stretch Fabric", "Casual Everyday Wear", "Athletic Performance Fit", "Fleece Lined", "Wrinkle-Free"]
    },
    "Home": {
        "brands": ["IKEA", "Dyson", "Philips", "Tupperware", "Bosch", "Cuisinart", "Ninja", "KitchenAid", "LG", "Whirlpool", "Breville", "Eufy"],
        "products": ["Blender", "Vacuum Cleaner", "Coffee Maker", "Bed Sheets", "Cookware Set", "Table Lamp", "Air Purifier", "Microwave Oven", "Robot Vacuum", "Toaster", "Stand Mixer", "Office Chair"],
        "features": ["Premium Stainless Steel", "Ceramic Non-Stick", "True HEPA Filter", "Energy Efficient Rating", "Egyptian Cotton King Size", "Smart LED Compatible", "Fully Programmable", "Dishwasher Safe", "Ergonomic Lumbar Support", "Quiet Operation"]
    },
    "Books": {
        "brands": ["Penguin", "HarperCollins", "Macmillan", "Simon & Schuster", "Scholastic", "Vintage", "Bantam", "O'Reilly"],
        "products": ["Novel", "Biography", "Cookbook", "Textbook", "Self-Help Book", "Sci-Fi Novel", "Mystery Thriller", "Poetry Collection", "Graphic Novel", "Historical Fiction", "Programming Guide"],
        "features": ["Hardcover Edition", "Mass Market Paperback", "Kindle eBook Edition", "New York Times Bestseller", "Fully Illustrated", "Revised and Updated Edition", "Pulitzer Prize Winning", "Audiobook Narration Included", "Special Collector's Edition", "Signed Copy"]
    },
    "Beauty": {
        "brands": ["L'Oreal", "MAC", "Clinique", "Maybelline", "Estee Lauder", "Dove", "Neutrogena", "Olay", "The Ordinary", "CeraVe", "Fenty Beauty", "NARS"],
        "products": ["Face Wash", "Moisturizer", "Liquid Lipstick", "Foundation", "Shampoo", "Perfume", "Sunscreen", "Eye Shadow Palette", "Vitamin C Serum", "Body Lotion", "Mascara", "Concealer"],
        "features": ["Broad Spectrum SPF 50", "Intensely Hydrating", "Matte Finish", "Anti-Aging Formula", "Sulfate and Paraben Free", "24-Hour Long Lasting", "Dermatologist Tested For Sensitive Skin", "100% Vegan & Cruelty-Free", "Fragrance-Free", "Infused with Hyaluronic Acid"]
    }
}

# Modifiers for extra realism
adjectives = ["Premium", "High-Quality", "Affordable", "Luxury", "Bestselling", "Top-Rated", "Exclusive", "Must-Have", "Elegant", "Durable", "Classic", "Modern"]

data = []
# Create exactly 700 items
for _ in range(700):
    category = random.choice(list(categories.keys()))
    brand = random.choice(categories[category]["brands"])
    product = random.choice(categories[category]["products"])
    feature = random.choice(categories[category]["features"])
    adj = random.choice(adjectives)
    
    # Variety of realistic listing titles/descriptions
    templates = [
        f"{brand} {product} - {feature}",
        f"{adj} {brand} {product}, {feature}",
        f"{brand} {adj} {product} with {feature}",
        f"Buy {brand} {product} ({feature})",
        f"{product} by {brand} | {feature} | {adj}",
        f"{adj} {product}: {brand} release featuring {feature}",
        f"{brand} Original {product} - {feature} - {adj} Choice",
        f"The {adj} {brand} {product} ({feature})"
    ]
    
    description = random.choice(templates)
    data.append([description, category])

# We can shuffle to randomize the order of categories
random.shuffle(data)

# Write to CSV
with open("data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(data)

print(f"Successfully generated {len(data)} product descriptions and saved to data.csv")
