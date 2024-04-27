import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# New dataset with 3 labels: historical, medical, and tax-related
historical_paragraphs = [
    "The Roman Empire was one of the largest and most powerful empires in history, spanning over three continents.",
    "The Renaissance period in Europe marked a cultural and artistic rebirth, characterized by advancements in art, literature, and science.",
    "The Industrial Revolution transformed societies from agrarian to industrial, leading to significant technological advancements and urbanization.",
    "The French Revolution of 1789 resulted in the overthrow of the monarchy and the establishment of a republic based on democratic principles.",
    "World War I, also known as the Great War, was a global conflict that lasted from 1914 to 1918 and involved many of the world's great powers.",
    "The American Civil War, fought between the Northern states (Union) and the Southern states (Confederacy) from 1861 to 1865, led to the abolition of slavery in the United States.",
    "The signing of the Magna Carta in 1215 marked a significant milestone in the establishment of constitutional rights and limitations on the power of the monarchy in England.",
    "The Age of Exploration, led by explorers such as Christopher Columbus and Vasco da Gama, expanded European influence and trade routes around the globe.",
    "The Fall of Constantinople in 1453 marked the end of the Byzantine Empire and the beginning of the Ottoman Empire's dominance in the region.",
    "The Treaty of Versailles, signed in 1919, officially ended World War I and imposed heavy penalties and territorial losses on Germany.",
    "The Enlightenment era in Europe fostered a wave of intellectual progress, emphasizing reason, individualism, and skepticism of traditional authority.",
    "The Scientific Revolution, spanning the 16th and 17th centuries, revolutionized the way humans understood the natural world through empirical evidence and scientific methodology.",
    "The discovery of the New World by Christopher Columbus in 1492 led to a dramatic encounter between the civilizations of Europe and the Americas, profoundly impacting the history of both continents.",
    "The rise of the Mongol Empire under Genghis Khan in the early 13th century created the largest contiguous land empire in history, spanning from Eastern Europe to the Sea of Japan.",
    "The Meiji Restoration in Japan (1868) marked the end of the Tokugawa shogunate and the beginning of a modernized, industrial Japan that would become a major world power.",
    "The Russian Revolution of 1917 led to the fall of the Tsarist autocracy and the rise of the Soviet Union, a major shift that would have lasting global implications throughout the 20th century.",
    "The Black Death, a devastating global epidemic of bubonic plague that struck Europe and Asia in the mid-14th century, killed an estimated 25 million people in Europe alone.",
    "The signing of the Declaration of Independence in 1776 marked the beginning of American self-governance and the revolutionary war against British rule.",
    "The discovery and subsequent popularization of penicillin in 1928 by Alexander Fleming introduced a new era of antibiotic treatment, revolutionizing medical practice.",
    "The Suez Crisis of 1956 was a pivotal event in the Cold War, highlighting the decline of British and French influence in the Middle East and the rising importance of superpower diplomacy.",
    "The Gold Rush of the mid-19th century significantly impacted the development of the American West, leading to rapid population growth and economic changes.",
    "The Cuban Missile Crisis of 1962 was a critical moment in the Cold War, bringing the United States and the Soviet Union close to nuclear conflict.",
    "The establishment of the United Nations in 1945 aimed to promote international cooperation and prevent future conflicts following the devastating effects of World War II.",
    "The abolition of the transatlantic slave trade in the British Empire in 1807 marked a crucial step towards ending the global slave trade.",
    "The signing of the Treaty of Waitangi in 1840 established British sovereignty over New Zealand, but also led to long-standing disputes over land rights and sovereignty with the Maori people.",
    "The Space Race, initiated in the late 1950s and culminating with the moon landing in 1969, was a symbol of technological and ideological rivalry between the USA and the Soviet Union.",
    "The partition of India in 1947 created two independent dominions, India and Pakistan, leading to significant migration and conflict in the region.",
    "The Cultural Revolution in China (1966-1976) was aimed at preserving communist ideology by purging remnants of capitalist and traditional elements from Chinese society.",
    "The fall of the Berlin Wall in 1989 symbolized the end of the Cold War and the reunification of East and West Germany in the following year.",
    "The signing of the Maastricht Treaty in 1992 led to the creation of the European Union, an economic and political union of multiple European states designed to foster economic cooperation."
]


medical_paragraphs = [
    "The discovery of penicillin in 1928 by Alexander Fleming marked the beginning of modern antibiotics, revolutionizing the treatment of bacterial infections.",
    "The development of the polio vaccine in the 1950s by Jonas Salk and later by Albert Sabin led to a dramatic reduction in cases of poliomyelitis worldwide.",
    "The completion of the Human Genome Project in 2003 provided a comprehensive map of all human genes, opening new avenues for medical research and personalized medicine.",
    "The introduction of antiseptic techniques by Joseph Lister in the 19th century significantly reduced infections during surgery, transforming surgical practice.",
    "The invention of the X-ray by Wilhelm Conrad Röntgen in 1895 revolutionized diagnostic medicine by allowing doctors to see inside a patient's body without surgery.",
    "The development of insulin therapy in the early 20th century by Frederick Banting and Charles Best provided a life-saving treatment for diabetes patients.",
    "The establishment of the World Health Organization in 1948 has played a crucial role in global health initiatives and the eradication of diseases like smallpox.",
    "The advent of antiretroviral therapy in the late 20th century has significantly extended the life expectancy of HIV/AIDS patients.",
    "The discovery of the structure of DNA in 1953 by James Watson and Francis Crick was pivotal in advancing the field of genetics and biotechnology.",
    "The introduction of anesthesia in the 1840s revolutionized surgery by allowing more complex procedures without causing pain to the patient.",
    "The development of chemotherapy in the 20th century provided a critical tool in the fight against cancer, offering hope to millions of patients.",
    "The creation of the first artificial heart in 1982 marked a significant advancement in medical devices, providing an option for patients awaiting heart transplants.",
    "The approval of the first recombinant DNA-derived vaccine, Hepatitis B, in the 1980s marked a significant step forward in immunization technology.",
    "The emergence of telemedicine has transformed patient care, making medical consultation accessible remotely, especially valuable in rural and underserved areas.",
    "The discovery of CRISPR-Cas9 gene editing technology has opened new possibilities for genetic engineering and the treatment of genetic disorders.",
    "The implementation of electronic health records has improved the efficiency of healthcare delivery by facilitating better data management and patient care coordination.",
    "The development of MRI technology in the 1970s provided a new, non-invasive method to view soft tissue, revolutionizing diagnostic medicine.",
    "The launch of public health campaigns, such as those against smoking, has significantly reduced rates of diseases related to lifestyle choices.",
    "The use of robotic surgery has enhanced the precision of surgical procedures, reducing recovery times and improving patient outcomes.",
    "The advancement in prosthetics using 3D printing technology has made these essential devices more accessible and customizable for patients.",
    "The introduction of targeted cancer therapies has provided treatments that are specific to the genetic makeup of a patient’s tumor, improving treatment outcomes.",
    "The development of the first successful blood transfusion in the 17th century set the foundation for modern transfusion medicine.",
    "The creation of digital health apps has empowered patients to manage their health more actively, providing tools for monitoring chronic conditions.",
    "The use of stem cell therapy has shown promise in regenerating damaged tissues and organs, offering new treatment avenues for previously incurable conditions.",
    "The establishment of medical ethics principles has guided the professional conduct of healthcare providers, ensuring patient rights and safety.",
    "The development of the smallpox vaccine by Edward Jenner in the late 18th century was the first successful attempt to control an infectious disease through vaccination.",
    "The rise of epidemiology as a key scientific field has been instrumental in controlling outbreaks and understanding the spread of diseases.",
    "The approval of immunotherapy treatments has marked a new era in the fight against cancer, enhancing the immune system’s ability to fight cancer cells.",
    "The pioneering of minimally invasive surgery in the 1980s has led to less traumatic surgical options for patients, promoting faster recovery and reduced hospital stays.",
    "The advancement of biodegradable implants offers a sustainable option for temporary support during tissue regeneration, reducing the need for additional surgeries."
]


tax_related_paragraphs = [
    "The Magna Carta, signed in 1215, was one of the first documents to put into writing the principle that the king and his government were not above the law.",
    "The establishment of the United States Constitution in 1787 created a federal system of government with checks and balances, fundamentally shaping American law and politics.",
    "The development of Common Law in England provided a foundation for legal systems in many English-speaking countries, emphasizing the role of court decisions in shaping law.",
    "The adoption of the Napoleonic Code in 1804 in France influenced the legal systems of many other countries by introducing a comprehensive system of written and accessible law.",
    "The Geneva Conventions, developed in a series of treaties from 1864 onward, set international standards for humanitarian treatment in war.",
    "The establishment of the International Court of Justice in 1945 provided a venue for resolving disputes between states under international law.",
    "The Miranda rights, established by the U.S. Supreme Court in 1966, require police to inform suspects of their rights upon arrest, including the right to remain silent.",
    "The concept of Legal Precedent, where past court decisions influence future cases, is a cornerstone of the judicial system in common law countries.",
    "Intellectual Property Law protects creators' rights to their inventions, designs, and artistic works to encourage innovation and creativity.",
    "Labor laws regulate the relationship between workers, employers, unions, and governments to ensure fair wages and safe working conditions.",
    "Environmental laws are designed to protect the environment from pollution and unsustainable practices by regulating air and water quality and waste management.",
    "Consumer protection laws prevent businesses from engaging in fraud or specified unfair practices to ensure fair trade for consumers.",
    "The Freedom of Information Act, enacted in 1966 in the U.S., allows for the full or partial disclosure of previously unreleased information controlled by the government.",
    "Anti-discrimination laws, such as the Civil Rights Act of 1964, prohibit discrimination based on race, color, religion, sex, or national origin.",
    "The Sarbanes-Oxley Act of 2002 was passed to protect investors from the possibility of fraudulent accounting activities by corporations.",
    "Cyberlaw deals with legal issues related to the use of information technology, including copyright, freedom of expression, and online privacy.",
    "Bankruptcy laws provide for the reduction or elimination of certain debts, and can provide a timeline for the repayment of non-dischargeable debts.",
    "Maritime Law governs nautical issues and private maritime disputes, such as shipping or offenses occurring on open water.",
    "Contract Law governs the legality of agreements made between two or more parties, ensuring that the agreements are enforceable in court.",
    "Tort Law addresses and provides remedies for civil wrongs not arising out of contractual obligations, such as negligence and defamation.",
    "Property Law governs the various forms of ownership and tenancy in real property and personal property within the common law legal system.",
    "Family Law deals with family-related issues and domestic relations including marriage, divorce, child custody, and domestic abuse.",
    "Criminal Law involves prosecution by the government of a person for an act that has been classified as a crime.",
    "Patent Law grants inventors exclusive rights to their inventions for a certain period in exchange for public disclosure of the invention.",
    "Securities Law covers the trading of securities and aims to protect investors from fraudulent schemes.",
    "Estate Law governs the disposition of a person’s estate after death, including wills, trusts, and probate procedures.",
    "Administrative Law covers the activities of administrative agencies of government including rulemaking, adjudication, and enforcement.",
    "Immigration Law governs who may enter, stay in, or need to leave a country, a matter regulated under international law.",
    "Antitrust Law, also known as competition law, is developed to protect consumers from predatory business practices and ensure fair competition.",
    "Tax Law involves the rules, policies, and laws that oversee the tax process, which involves charges on estates, transactions, property, income, licenses and more."
]


# Combine paragraphs and labels
paragraphs = historical_paragraphs + medical_paragraphs + tax_related_paragraphs
# Assuming you have 29 historical, 30 medical, and 30 tax-related paragraphs
labels = ["historical"] * 30 + ["medical"] * 30 + ["tax-related"] * 30

# Rest of the script remains the same
tokenizer = Tokenizer()
tokenizer.fit_on_texts(paragraphs)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences and pad them to ensure uniform input size
sequences = tokenizer.texts_to_sequences(paragraphs)
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert labels to one-hot encoding
label_dict = {'historical': 0, 'medical': 1, 'tax-related': 2}
labels = [label_dict[label] for label in labels]
one_hot_labels = np.zeros((len(labels), len(label_dict)))
for i, label in enumerate(labels):
    one_hot_labels[i, label] = 1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, one_hot_labels, test_size=0.3, random_state=42)

# Build the CNN model
embedding_dim = 100
filters = 128
kernel_size = 5

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_dict), activation='softmax'))

from tensorflow.keras.optimizers import Adam

# Set a specific learning rate
learning_rate = 0.0001  # You can adjust this value

model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')
