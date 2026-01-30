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

// all color settings

// dark blue
// sf::Color bgColor(15, 23, 42);
// sf::Color panel(2, 6, 23);
// sf::Color primary(56, 189, 248);
// sf::Color secondary(34, 211, 238);
// sf::Color texts(220, 227, 237);
// sf::Color borders(30, 41, 59);

// light green
// sf::Color bgColor(247, 251, 250);
// sf::Color panel(255, 255, 255);
// sf::Color primary(45, 212, 191);
// sf::Color secondary(94, 234, 212);
// sf::Color texts(19, 78, 74);
// sf::Color borders(204, 251, 241);

// hacker green
sf::Color bgColor(0,0,0);
sf::Color panel(0,0,0);
sf::Color primary(97,207,90);
sf::Color secondary(97,207,90);
sf::Color textprimary(97,207,90);
sf::Color textsecondary(97,207,90);

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

int main() {
    // Neural network setup
    string model_name = "784_100_10";
    Network net({0});
    net.load("model/model_data_" + model_name + ".txt");

    // Window setup
    sf::RenderWindow window(VideoMode({WINDOW_WIDTH, WINDOW_HEIGHT}),
                            "AI Digit Reader");
    window.setFramerateLimit(FPS);
    window.setView(window.getDefaultView());

    // The Drawing canvas
    float canvas_x = WINDOW_WIDTH / 2 - WIDTH / 2;
    float canvas_y = 56;
    sf::RectangleShape paper;
    paper.setSize({WIDTH, HEIGHT});
    paper.setFillColor(panel);
    paper.setOutlineThickness(1.f);
    paper.setOutlineColor(textsecondary);  // Subtle border
    paper.setPosition({canvas_x, canvas_y});

    // Inline of canvas
    sf::RectangleShape outline;
    outline.setSize(
        {static_cast<float>(WIDTH - 72), static_cast<float>(HEIGHT - 72)});
    outline.setFillColor(panel);
    outline.setOutlineThickness(1.f);
    outline.setOutlineColor(secondary);  // Subtle border
    outline.setPosition({static_cast<float>(canvas_x + 36),
                         static_cast<float>(canvas_y + 36)});

    // Canvas pixels setup=
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
    result.setFillColor(textprimary);
    result.setStyle(sf::Text::Bold);
    result.setPosition({static_cast<float>(WINDOW_WIDTH / 2 - 90),
                        static_cast<float>(canvas_y + HEIGHT + 28)});
    result.setString("Result : ");

    sf::Text desc(font);
    desc.setString("Write a digit inside the box to try the ai !");
    desc.setCharacterSize(18);
    desc.setFillColor(textsecondary);
    desc.setStyle(sf::Text::Bold);
    desc.setPosition({static_cast<float>(WINDOW_WIDTH / 2 - 190), 10});

    // cursor related
    uint64_t mouseOff = 0;
    uint64_t frameCount = 0;
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
                    print_eval(net.get_eval(), net.get_value());
                }
            }
        }

        if (Mouse::isButtonPressed(Mouse::Button::Left)) {
            Vector2i pos = Mouse::getPosition(window);
            int x = (pos.x - canvas_x) / PIXEL;
            int y = (pos.y - canvas_y) / PIXEL;

            if (x >= 0 && x < WIDTH_P && y >= 0 && y < HEIGHT_P) {
                pixels[y * HEIGHT_P + x] = 1.0;
                mouseOff = 0;
                readActive = true;
                if (frameCount % 6 == 0) {
                    net.evaluate(pixels);
                }
                
            }
        }

        // if stopped drawing, read the number
        if (mouseOff > FPS * 0.2 && readActive) {
            net.evaluate(pixels);
            result.setString("Result : " + to_string(net.get_value()));
            readActive = false;
        }

        mouseOff++;
        frameCount++;

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
                block.setFillColor(primary);
                window.draw(block);
            }
        }

        for (int i = 0; i < net.get_eval().size(); i++) {
            RectangleShape block(
                Vector2f({36, static_cast<float>(36 * net.get_eval()[i])}));
            block.setPosition(
                {static_cast<float>(eval_x + i * 44.8 + 2),
                 static_cast<float>(eval_y + 2 + 36 * (1 - net.get_eval()[i]))});
            block.setFillColor(primary);
            RectangleShape frame(Vector2f({40, 40}));
            frame.setPosition({static_cast<float>(eval_x + i * 44.8),
                               static_cast<float>(eval_y)});
            frame.setFillColor(bgColor);
            frame.setOutlineThickness(1.f);
            frame.setOutlineColor(secondary);
            window.draw(frame);
            window.draw(block);
            sf::Text digit(font);
            digit.setString(to_string(i));
            digit.setCharacterSize(18);
            digit.setFillColor(textprimary);
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
