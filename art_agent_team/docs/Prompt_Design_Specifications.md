# Prompt Design Specifications

## Master Vision Prompt

Analyze the image and filename tokens provided. Extract the following fields:
1. author
2. title
3. date
4. nationality
5. style (from the predefined list: {style_list})
6. movement (from the predefined list: 
* Renaissance (c. 1400–1527)
* Mannerism (1520s–1600)
* Baroque (1600–1725)
* Dutch Golden Age (1585–1702)
* Rococo (1720–1780)
* Neoclassicism (1750–1840)
* Romanticism (1800–1850)
* Realism (1840–1870)
* Pre-Raphaelite Brotherhood (1848–1854)
* Barbizon School (1830–1870)
* Hudson River School (1826–1870, US)
* Luminism (1850s–1870s, US)
* Academic Art (c. 1840–1900)
* Impressionism (1870–1900)
* Post-Impressionism (1886–1905)
* Symbolism (1886–1900)
* Naturalism (1880–1900)
* Art Nouveau (1895–1915)
* Fauvism (1905–1910)
* Expressionism (1890–1939)
* Cubism (1905–1939)
* Futurism (1909–1918)
* Dadaism (1912–1923)
* Constructivism (1913–1930)
* Precisionism (1920–1950, US)
* Bauhaus (1920–1925)
* Surrealism (1924–1945)
* De Stijl (1917–1931)
* New Objectivity (1918–1933)
* Harlem Renaissance (1920–1930, US)
* Art Deco (1920–1935)
* Abstract Expressionism (1945–1960, US)
* Color Field Painting (late 1940s–1960s, US)
* Action Painting (1940s–1950s, US)
* Hard-Edge Painting (1950s–1960s)
* Pop Art (1956–1969)
* Op Art (1965–1970)
* Minimalism (1960–1975)
* Photorealism (1968–present)
* Pop Surrealism (1970–present)
* Arte Povera (1960–1969)
* Neo-Expressionism (late 1970s–1990s)
* Street Art (1980s–present)
* Contemporary Art (1978–present)
)
7. primary_subjects
8. secondary_subjects
9. brief_description
10. confidence_score
11. grounding_used
12. limitations

Validate the identified 'date' against the time range associated with the identified 'movement' from the provided list. Ensure the date falls within the movement's time range.

Output the results in JSON format only.

Additional instructions for the model: {model_specific_instructions}

Filename tokens: {filename_tokens}

## Consolidation Prompt

Consolidate the results from multiple vision models provided as a JSON string. Reference the 'movement' field and use the provided movement list (
* Renaissance (c. 1400–1527)
* Mannerism (1520s–1600)
* Baroque (1600–1725)
* Dutch Golden Age (1585–1702)
* Rococo (1720–1780)
* Neoclassicism (1750–1840)
* Romanticism (1800–1850)
* Realism (1840–1870)
* Pre-Raphaelite Brotherhood (1848–1854)
* Barbizon School (1830–1870)
* Hudson River School (1826–1870, US)
* Luminism (1850s–1870s, US)
* Academic Art (c. 1840–1900)
* Impressionism (1870–1900)
* Post-Impressionism (1886–1905)
* Symbolism (1886–1900)
* Naturalism (1880–1900)
* Art Nouveau (1895–1915)
* Fauvism (1905–1910)
* Expressionism (1890–1939)
* Cubism (1905–1939)
* Futurism (1909–1918)
* Dadaism (1912–1923)
* Constructivism (1913–1930)
* Precisionism (1920–1950, US)
* Bauhaus (1920–1925)
* Surrealism (1924–1945)
* De Stijl (1917–1931)
* New Objectivity (1918–1933)
* Harlem Renaissance (1920–1930, US)
* Art Deco (1920–1935)
* Abstract Expressionism (1945–1960, US)
* Color Field Painting (late 1940s–1960s, US)
* Action Painting (1940s–1950s, US)
* Hard-Edge Painting (1950s–1960s)
* Pop Art (1956–1969)
* Op Art (1965–1970)
* Minimalism (1960–1975)
* Photorealism (1968–present)
* Pop Surrealism (1970–present)
* Arte Povera (1960–1969)
* Neo-Expressionism (late 1970s–1990s)
* Street Art (1980s–present)
* Contemporary Art (1978–present)
) to validate the consolidated 'date' against the consolidated 'movement''s time range. If there are conflicts or low confidence, choose the most plausible movement.

Output the consolidated results in JSON format only.