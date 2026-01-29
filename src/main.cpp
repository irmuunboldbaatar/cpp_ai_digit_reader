#include <SFML/Graphics.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "network.h"
using namespace std;
using namespace sf;

// canvas config
float PIXEL = 8.0;
int WIDTH_P = 28;
int HEIGHT_P = 28;
int TOTAL_PIXEL = WIDTH_P * HEIGHT_P;
float WIDTH = WIDTH_P * PIXEL;    // = 224  when PIXEL = 8.0 WIDTH_P = 28
float HEIGHT = HEIGHT_P * PIXEL;  // = 224  when PIXEL = 8.0 HEIGHT_P = 28

// window config
int FPS = 120;
unsigned int WINDOW_WIDTH =
    WIDTH + 280;  // = 504  when PIXEL = 8.0 WIDTH_P = 28
unsigned int WINDOW_HEIGHT =
    HEIGHT + 280;  // = 448  when PIXEL = 8.0 HEIGHT_P = 28

void print_eval(vector<double> eval, int value) {
    cout << "Evaluation:" << endl;
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
    cout << "Result : " + to_string(value) << endl;
}

void drawThickLine(sf::RenderWindow& window, sf::Vector2f point1, sf::Vector2f point2, float thickness, sf::Color color) {
    sf::Vector2f direction = point2 - point1;
    float length = std::sqrt(direction.x * direction.x + direction.y * direction.y);
    
    if (length > 0) {
        sf::RectangleShape line(sf::Vector2f(length, thickness));
        line.setOrigin({0, thickness / 2.f});
        line.setPosition(point1);
        line.setRotation(sf::degrees(std::atan2(direction.y, direction.x) * 180.f / 3.14159f));
        line.setFillColor(color);
        window.draw(line);
        
        // Add rounded caps for smoothness
        sf::CircleShape cap(thickness / 2.f);
        cap.setOrigin({thickness / 2.f, thickness / 2.f});
        cap.setFillColor(color);
        cap.setPosition(point1);
        window.draw(cap);
        cap.setPosition(point2);
        window.draw(cap);
    }
}

int main() {
    // Neural network setup
    string model_name = "784_100_10";
    Network net({0});
    net.load("model/model_data_" + model_name + ".txt");

    // Window setup
    sf::Color bgColor(241, 245, 249);
    sf::RenderWindow window(VideoMode({WINDOW_WIDTH, WINDOW_HEIGHT}),
                            "AI Digit Reader");
    window.setFramerateLimit(FPS);
    window.setView(window.getDefaultView());

    // The Drawing canvas
    float canvas_x = WINDOW_WIDTH / 2 - WIDTH / 2;
    float canvas_y = 56;
    sf::RectangleShape paper;
    paper.setSize({WIDTH, HEIGHT});
    paper.setFillColor(sf::Color::White);
    paper.setOutlineThickness(1.f);
    paper.setOutlineColor(sf::Color(226, 232, 240));  // Subtle border
    paper.setPosition({canvas_x, canvas_y});

    // Inline of canvas
    sf::RectangleShape outline;
    outline.setSize(
        {static_cast<float>(WIDTH - 72), static_cast<float>(HEIGHT - 72)});
    outline.setFillColor(sf::Color::White);
    outline.setOutlineThickness(1.f);
    outline.setOutlineColor(sf::Color(226, 232, 240));  // Subtle border
    outline.setPosition({static_cast<float>(canvas_x + 36),
                         static_cast<float>(canvas_y + 36)});

    // Canvas pixels setup
    sf::Color inkColor(51, 65, 85);
    vector<double> pixels(TOTAL_PIXEL, 0.0);

    // Create and configure the text
    sf::Font font;
    if (!font.openFromFile("assets/notosans.ttf")) {
        std::cerr << "Error: Could not load font!" << std::endl;
        return -1;
    }

    // Evalutation display
    int eval_x = WINDOW_WIDTH / 2 - 224;
    int eval_y = canvas_y + HEIGHT + 28 + 40 + 28;

    sf::Text result(font);
    result.setCharacterSize(40);
    result.setFillColor(sf::Color(71, 85, 105));
    result.setStyle(sf::Text::Bold);
    result.setPosition({static_cast<float>(WINDOW_WIDTH / 2 - 90),
                        static_cast<float>(canvas_y + HEIGHT + 28)});
    result.setString("Result : ");

    sf::Text desc(font);
    desc.setString("Write a digit inside the box to try the ai !");
    desc.setCharacterSize(18);
    desc.setFillColor(sf::Color(71, 85, 105));
    desc.setStyle(sf::Text::Bold);
    desc.setPosition({static_cast<float>(WINDOW_WIDTH / 2 - 190), 10});

    // cursor related
    uint64_t mouseOff = 0;
    bool readActive = false;

    sf::Vector2f lastPos;
    bool isDrawing = false;
    std::vector<std::pair<sf::Vector2f, sf::Vector2f>> lines;

    while (window.isOpen()) {
        while (const optional event = window.pollEvent()) {
            if (event->is<Event::Closed>()) window.close();

            if (const auto* keyPressed =
                    event->getIf<sf::Event::KeyPressed>()) {
                if (keyPressed->code == sf::Keyboard::Key::Space ||
                    keyPressed->code == sf::Keyboard::Key::Backspace) {
                    for (int i = 0; i < TOTAL_PIXEL; i++) pixels[i] = 0;
                    cout << "Canvas cleared." << endl;
                }

                if (keyPressed->code == sf::Keyboard::Key::Enter) {
                    print_eval(net.getEval(), net.getValue());
                }
            }
        }

        if (Mouse::isButtonPressed(Mouse::Button::Left)) {
            Vector2i pos = Mouse::getPosition(window);
            int x = (pos.x - canvas_x) / PIXEL;
            int y = (pos.y - canvas_y) / PIXEL;

            if (x >= 0 && x < WIDTH_P && y >= 0 && y < HEIGHT_P) {
                pixels[y * HEIGHT_P + x] = 1.0;
                net.evaluate(pixels);
                mouseOff = 0;
                readActive = true;
            }
        }

        // if stopped drawing, read the number
        if (mouseOff > FPS * 0.2 && readActive) {
            net.evaluate(pixels);
            result.setString("Result : " + to_string(net.getValue()));
            readActive = false;
        }

        mouseOff++;

        window.clear(bgColor);
        window.draw(result);
        window.draw(desc);
        window.draw(paper);
        window.draw(outline);

        for (int i = 0; i < TOTAL_PIXEL; i++) {
            if (pixels[i] > 0) {
                RectangleShape block(Vector2f(
                    {static_cast<float>(PIXEL), static_cast<float>(PIXEL)}));
                block.setPosition(
                    {static_cast<float>((i % WIDTH_P) * PIXEL + canvas_x),
                     static_cast<float>((i / HEIGHT_P) * PIXEL + canvas_y)});
                block.setFillColor(inkColor);
                window.draw(block);
            }
        }

        for (int i = 0; i < net.getEval().size(); i++) {
            RectangleShape block(
                Vector2f({36, static_cast<float>(36 * net.getEval()[i])}));
            block.setPosition(
                {static_cast<float>(eval_x + i * 44.8 + 2),
                 static_cast<float>(eval_y + 2 + 36 * (1 - net.getEval()[i]))});
            block.setFillColor(inkColor);
            RectangleShape frame(Vector2f({40, 40}));
            frame.setPosition({static_cast<float>(eval_x + i * 44.8),
                               static_cast<float>(eval_y)});
            frame.setFillColor(bgColor);
            frame.setOutlineThickness(1.f);
            frame.setOutlineColor(inkColor);
            window.draw(frame);
            window.draw(block);
            sf::Text digit(font);
            digit.setString(to_string(i));
            digit.setCharacterSize(18);
            digit.setFillColor(sf::Color(71, 85, 105));
            digit.setStyle(sf::Text::Bold);
            digit.setPosition({static_cast<float>(eval_x + i * 44.8 + 14),
                               static_cast<float>(eval_y + 40 + 14)});
            window.draw(digit);
        }

        window.display();
    }

    cout << "Window closed." << endl;
    return 0;
}
