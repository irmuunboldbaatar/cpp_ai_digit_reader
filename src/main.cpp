#include <SFML/Graphics.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "network.h"
using namespace std;
using namespace sf;

float PIXEL = 8.0;
int WIDTH_P = 28;
int HEIGHT_P = 28;

int TOTAL_PIXEL = WIDTH_P * HEIGHT_P;
int WIDTH = WIDTH_P * PIXEL;
int HEIGHT = HEIGHT_P * PIXEL;

bool isEmpty(vector<double> v) {
    for (int i = 0; i < v.size(); i++)
        if (v[i] != 0) return false;
    return true;
}

int main() {
    string model_name = "784_100_10";
    // Neural network setup
    Network net({0});
    net.load("model/model_data_" + model_name + ".txt");

    // Window setup
    RenderWindow window(VideoMode({448, 224}), "AI Digit Reader");
    window.setFramerateLimit(120);
    window.setView(window.getDefaultView());

    vector<double> pixels(TOTAL_PIXEL, 0.0);

    // Create and configure the text
    sf::Font font;
    if (!font.openFromFile("assets/notosans.ttf")) {
        std::cerr << "Error: Could not load font!" << std::endl;
        return -1;
    }
    sf::Text text(font);
    text.setString("Read :");
    text.setCharacterSize(40);  // in pixels
    text.setFillColor(sf::Color::White);
    text.setStyle(sf::Text::Bold);
    text.setPosition({234, 60});
    sf::Text desc(font);
    // desc.setString(
    //     "Durvuljin dotor golluulan\nneg tsifr bichij\nEnter darna
    //     uu.\n\n\n\n\n\nEnter  : " "Unshuulah\nSpace : Arilgah\n\nmodel id:"
    //     + model_name);
    desc.setString(
        "Write a digit\nneatly in the box\nso that it fits in "
        "it.\n\n\n\n\n\nEnter  : "
        "make ai read digit\nSpace : clear box\n\nmodel id:" +
        model_name);
    desc.setCharacterSize(15);  // in pixels
    desc.setFillColor(sf::Color::White);
    desc.setStyle(sf::Text::Bold);
    desc.setPosition({234, 5});

    std::array line1 = {sf::Vertex{sf::Vector2f(36, 36)},
                        Vertex{sf::Vector2f(36, 192)}};
    std::array line2 = {sf::Vertex{sf::Vector2f(36, 36)},
                        Vertex{sf::Vector2f(192, 36)}};
    std::array line3 = {sf::Vertex{sf::Vector2f(192, 192)},
                        Vertex{sf::Vector2f(192, 36)}};
    std::array line4 = {sf::Vertex{sf::Vector2f(192, 192)},
                        Vertex{sf::Vector2f(36, 192)}};

    while (window.isOpen()) {
        // SFML 3 uses a new event polling system
        while (const optional event = window.pollEvent()) {
            if (event->is<Event::Closed>()) window.close();

            if (const auto* keyPressed =
                    event->getIf<sf::Event::KeyPressed>()) {
                if (keyPressed->code == sf::Keyboard::Key::Space) {
                    for (int i = 0; i < TOTAL_PIXEL; i++) pixels[i] = 0;
                    cout << "Canvas cleared." << endl;
                }

                if (keyPressed->code == sf::Keyboard::Key::Enter) {
                    net.evaluate(pixels);
                    cout << "Evaluation:" << endl;
                    vector<double> eval = net.getEval();
                    for (int i = 0; i < eval.size(); i++) {
                        // Create the bar: 1 character for every 2% of
                        // confidence
                        int barWidth = static_cast<int>(eval[i] * 100 / 2.0);
                        string bar = "";
                        for (int b = 0; b < barWidth; b++) bar += "█";

                        // Print formatted line: "Digit X : [Percent]% [Bar]"
                        printf("Digit %d : %6.2f%% ", i, eval[i] * 100);
                        cout << bar << endl;
                    }
                    cout << "Read : " + to_string(net.getValue()) << endl;
                    text.setString("Read : " + to_string(net.getValue()));
                }
            }
        }

        if (Mouse::isButtonPressed(Mouse::Button::Left)) {
            Vector2i pos = Mouse::getPosition(window);
            int x = pos.x / PIXEL;
            int y = pos.y / PIXEL;

            if (x >= 0 && x < WIDTH_P && y >= 0 && y < HEIGHT_P) {
                pixels[y * HEIGHT_P + x] = 1.0;
            }
        }

        window.clear(Color(30, 30, 30));
        window.draw(text);
        window.draw(desc);
        RectangleShape draw_bg(Vector2f(
            {static_cast<float>(WIDTH), static_cast<float>(HEIGHT)}));
        // In SFML 3, setPosition takes a Vector2f
        draw_bg.setPosition({0.0, 0.0});
        draw_bg.setFillColor(Color::Black);
        window.draw(draw_bg);
        for (int i = 0; i < TOTAL_PIXEL; i++) {
            if (pixels[i] > 0) {
                RectangleShape block(Vector2f(
                    {static_cast<float>(PIXEL), static_cast<float>(PIXEL)}));
                // In SFML 3, setPosition takes a Vector2f
                block.setPosition(
                    {static_cast<float>((i % WIDTH_P) * PIXEL),
                     static_cast<float>((i / HEIGHT_P) * PIXEL)});
                block.setFillColor(Color::White);
                window.draw(block);
            }
        }

        window.draw(line1.data(), line1.size(), sf::PrimitiveType::Lines);
        window.draw(line2.data(), line2.size(), sf::PrimitiveType::Lines);
        window.draw(line3.data(), line3.size(), sf::PrimitiveType::Lines);
        window.draw(line4.data(), line4.size(), sf::PrimitiveType::Lines);

        window.display();
    }

    cout << "Window closed." << endl;
    return 0;
}
