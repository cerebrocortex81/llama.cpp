#include "json-schema-to-grammar.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::json;
using namespace std;

const string SPACE_RULE = "\" \"?";

unordered_map<string, string> PRIMITIVE_RULES = {
    {"boolean", "(\"true\" | \"false\") space"},
    {"number", "(\"-\"? ([0-9] | [1-9] [0-9]*)) (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)? space"},
    {"integer", "(\"-\"? ([0-9] | [1-9] [0-9]*)) space"},
    {"value", "object | array | string | number | boolean"},
    {"object", "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space"},
    {"array", "\"[\" space ( value (\",\" space value)* )? \"]\" space"},
    {"uuid", "\"\\\"\" [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F] "
                "\"-\" [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F] "
                "\"-\" [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F] "
                "\"-\" [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F] "
                "\"-\" [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F] \"\\\"\" space"},
    {"string", " \"\\\"\" (\n"
               "        [^\"\\\\] |\n"
               "        \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])\n"
               "      )* \"\\\"\" space"},
    {"null", "\"null\" space"}
};
vector<string> OBJECT_RULE_NAMES = {"object", "array", "string", "number", "boolean", "null", "value"};

unordered_map<string, string> DATE_RULES = {
    {"date", "[0-9] [0-9] [0-9] [0-9] \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )"},
    {"time", "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9] [0-9] [0-9] )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )"},
    {"date-time", "date \"T\" time"},
    {"date-string", "\"\\\"\" date \"\\\"\" space"},
    {"time-string", "\"\\\"\" time \"\\\"\" space"},
    {"date-time-string", "\"\\\"\" date-time \"\\\"\" space"}
};

static bool is_reserved_name(const string& name) {
    static std::unordered_set<std::string> RESERVED_NAMES;
    if (RESERVED_NAMES.empty()) {
        RESERVED_NAMES.insert("root");
        for (const auto &p : PRIMITIVE_RULES) RESERVED_NAMES.insert(p.first);
        for (const auto &p : DATE_RULES) RESERVED_NAMES.insert(p.first);
    }
    return RESERVED_NAMES.find(name) != RESERVED_NAMES.end();
}

regex INVALID_RULE_CHARS_RE("[^a-zA-Z0-9-]+");
regex GRAMMAR_LITERAL_ESCAPE_RE("[\r\n\"]");
regex GRAMMAR_RANGE_LITERAL_ESCAPE_RE("[\r\n\"\\]\\-\\\\]");
unordered_map<char, string> GRAMMAR_LITERAL_ESCAPES = {
    {'\r', "\\r"}, {'\n', "\\n"}, {'"', "\\\""}, {'-', "\\-"}, {']', "\\]"}
};

unordered_set<char> NON_LITERAL_SET = {'|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?'};
unordered_set<char> ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = {'[', ']', '(', ')', '|', '{', '}', '*', '+', '?'};

template <typename Iterator>
string join(Iterator begin, Iterator end, const string& separator) {
    ostringstream result;
    if (begin != end) {
        result << *begin;
        for (Iterator it = begin + 1; it != end; ++it) {
            result << separator << *it;
        }
    }
    return result.str();
}

static std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    tokens.push_back(str.substr(start));

    return tokens;
}

static string repeat(const string& str, size_t n) {
    if (n == 0) {
        return "";
    }

    string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

static std::string replacePattern(const std::string& input, const regex& regex, const function<string(const smatch &)>& replacement) {
    std::smatch match;
    std::string result;

    std::string::const_iterator searchStart(input.cbegin());
    std::string::const_iterator searchEnd(input.cend());

    while (std::regex_search(searchStart, searchEnd, match, regex)) {
        result.append(searchStart, searchStart + match.position());
        result.append(replacement(match));
        searchStart = match.suffix().first;
    }

    result.append(searchStart, searchEnd);

    return result;
}

static string _format_literal(const string& literal) {
    string escaped = replacePattern(json(literal).dump(), GRAMMAR_LITERAL_ESCAPE_RE, [&](const smatch& match) {
        char c = match.str()[0];
        return GRAMMAR_LITERAL_ESCAPES.at(c);
    });
    return "\"" + escaped + "\"";
}

static string _format_range_char(const string& ch) {
    return replacePattern(ch, GRAMMAR_RANGE_LITERAL_ESCAPE_RE, [&](const smatch& match) {
        char c = match.str()[0];
        return GRAMMAR_LITERAL_ESCAPES.at(c);
    });
}


class SchemaConverter {
private:
    std::function<json(const string&)> _fetch_json;
    bool _dotall;
    map<string, string> _rules;
    unordered_map<string, nlohmann::json> _refs;
    unordered_set<string> _refs_being_resolved;
    vector<string> _errors;
    vector<string> _warnings;

    string _add_rule(const string& name, const string& rule) {
        string esc_name = regex_replace(name, INVALID_RULE_CHARS_RE, "-");
        if (_rules.find(esc_name) == _rules.end() || _rules[esc_name] == rule) {
            _rules[esc_name] = rule;
            return esc_name;
        } else {
            int i = 0;
            while (_rules.find(esc_name + to_string(i)) != _rules.end() && _rules[esc_name + to_string(i)] != rule) {
                i++;
            }
            string key = esc_name + to_string(i);
            _rules[key] = rule;
            return key;
        }
    }

    string _generate_union_rule(const string& name, const vector<json>& alt_schemas) {
        vector<string> rules;
        for (size_t i = 0; i < alt_schemas.size(); i++) {
            rules.push_back(visit(alt_schemas[i], name + (name.empty() ? "alternative-" : "-") + to_string(i)));
        }
        return join(rules.begin(), rules.end(), " | ");
    }

    string _visit_pattern(const string& pattern, const string& name) {
        if (!(pattern.front() == '^' && pattern.back() == '$')) {
            _errors.push_back("Pattern must start with '^' and end with '$'");
            return "";
        }
        string sub_pattern = pattern.substr(1, pattern.length() - 2);
        unordered_map<string, string> sub_rule_ids;

        size_t i = 0;
        size_t length = sub_pattern.length();

        using literal_or_rule = pair<string, bool>;
        auto to_rule = [&](const literal_or_rule& ls) {
            auto is_literal = ls.second;
            auto s = ls.first;
            return is_literal ? "\"" + s + "\"" : s;
        };
        std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
            size_t start = i;
            vector<literal_or_rule> seq;

            auto get_dot = [&]() {
                string rule;
                if (_dotall) {
                    rule = "[\\U00000000-\\U0010FFFF]";
                } else {
                    rule = "[\\U00000000-\\x09\\x0B\\x0C\\x0E-\\U0010FFFF]";
                }
                return _add_rule("dot", rule);
            };

            // Joins the sequence, merging consecutive literals together.
            auto join_seq = [&]() {
                vector<literal_or_rule> ret;

                string literal;
                auto flush_literal = [&]() {
                    if (literal.empty()) {
                        return false;
                    }
                    ret.push_back(make_pair(literal, true));
                    literal.clear();
                    return true;
                };

                for (const auto& item : seq) {
                    auto is_literal = item.second;
                    if (is_literal) {
                        literal += item.first;
                    } else {
                        flush_literal();
                        ret.push_back(item);
                    }
                }
                flush_literal();

                vector<string> results;
                for (const auto& item : ret) {
                    results.push_back(to_rule(item));
                }
                return make_pair(join(results.begin(), results.end(), " "), false);
            };

            while (i < length) {
                char c = sub_pattern[i];
                if (c == '.') {
                    seq.push_back(make_pair(get_dot(), false));
                    i++;
                } else if (c == '(') {
                    i++;
                    if (i < length) {
                        if (sub_pattern[i] == '?') {
                            _warnings.push_back("Unsupported pattern syntax");
                        }
                    }
                    seq.push_back(make_pair("(" + to_rule(transform()) + ")", false));
                } else if (c == ')') {
                    i++;
                    if (start > 0 && sub_pattern[start - 1] != '(') {
                        _errors.push_back("Unbalanced parentheses");
                    }
                    return join_seq();
                } else if (c == '[') {
                    string square_brackets = string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != ']') {
                        if (sub_pattern[i] == '\\') {
                            square_brackets += sub_pattern.substr(i, 2);
                            i += 2;
                        } else {
                            square_brackets += sub_pattern[i];
                            i++;
                        }
                    }
                    if (i >= length) {
                        _errors.push_back("Unbalanced square brackets");
                    }
                    square_brackets += ']';
                    i++;
                    seq.push_back(make_pair(square_brackets, false));
                } else if (c == '|') {
                    seq.push_back(make_pair("|", false));
                    i++;
                } else if (c == '*' || c == '+' || c == '?') {
                    seq.back() = make_pair(to_rule(seq.back()) + c, false);
                    i++;
                } else if (c == '{') {
                    string curly_brackets = string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != '}') {
                        curly_brackets += sub_pattern[i];
                        i++;
                    }
                    if (i >= length) {
                        _errors.push_back("Unbalanced curly brackets");
                    }
                    curly_brackets += '}';
                    i++;
                    auto nums = split(curly_brackets.substr(1, curly_brackets.length() - 2), ",");
                    int min_times = 0;
                    int max_times = numeric_limits<int>::max();
                    try {
                        if (nums.size() == 1) {
                            min_times = max_times = std::stoi(nums[0]);
                        } else if (nums.size() != 2) {
                            _errors.push_back("Wrong number of values in curly brackets");
                        } else {
                            if (!nums[0].empty()) {
                                min_times = std::stoi(nums[0]);
                            }
                            if (!nums[1].empty()) {
                                max_times = std::stoi(nums[1]);
                            }
                        }
                    } catch (const std::invalid_argument& e) {
                        _errors.push_back("Invalid number in curly brackets");
                        return make_pair("", false);
                    }
                    auto &last = seq.back();
                    auto &sub = last.first;
                    auto sub_is_literal = last.second;

                    if (min_times == 0 && max_times == numeric_limits<int>::max()) {
                        sub += "*";
                    } else if (min_times == 0 && max_times == 1) {
                        sub += "?";
                    } else if (min_times == 1 && max_times == numeric_limits<int>::max()) {
                        sub += "+";
                    } else {
                        if (!sub_is_literal) {
                            string& sub_id = sub_rule_ids[sub];
                            if (sub_id.empty()) {
                                sub_id = _add_rule(name + "-" + to_string(sub_rule_ids.size()), sub);
                            }
                            sub = sub_id;
                        }
                        string result;
                        if (sub_is_literal && min_times > 0) {
                            result = "\"" + repeat(sub.substr(1, sub.length() - 2), min_times) + "\"";
                        } else {
                            for (int j = 0; j < min_times; j++) {
                                if (j > 0) {
                                    result += " ";
                                }
                                result += sub;
                            }
                        }
                        if (min_times > 0 && min_times < max_times) {
                            result += " ";
                        }
                        if (max_times == numeric_limits<int>::max()) {
                            result += sub + "*";
                        } else {
                            for (int j = min_times; j < max_times; j++) {
                                if (j > min_times) {
                                    result += " ";
                                }
                                result += sub + "?";
                            }
                        }
                        seq.back().first = result;
                        seq.back().second = false;
                    }
                } else {
                    string literal;
                    auto is_non_literal = [&](char c) {
                        return NON_LITERAL_SET.find(c) != NON_LITERAL_SET.end();
                    };
                    while (i < length) {
                        if (sub_pattern[i] == '\\' && i < length - 1) {
                            char next = sub_pattern[i + 1];
                            if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.find(next) != ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.end()) {
                                i++;
                                literal += sub_pattern[i];
                                i++;
                            } else {
                                literal += sub_pattern.substr(i, 2);
                                i += 2;
                            }
                        } else if (sub_pattern[i] == '"') {
                            literal += "\\\"";
                            i++;
                        } else if (!is_non_literal(sub_pattern[i]) &&
                                (i == length - 1 || literal.empty() || sub_pattern[i + 1] == '.' || !is_non_literal(sub_pattern[i + 1]))) {
                            literal += sub_pattern[i];
                            i++;
                        } else {
                            break;
                        }
                    }
                    if (!literal.empty()) {
                        seq.push_back(make_pair(literal, true));
                    }
                }
            }
            return join_seq();
        };
        return _add_rule(name, "\"\\\"\" " + to_rule(transform()) + " \"\\\"\" space");
    }

    string _resolve_ref(const string& ref) {
        string ref_name = ref.substr(ref.find_last_of('/') + 1);
        if (_rules.find(ref_name) == _rules.end() && _refs_being_resolved.find(ref) == _refs_being_resolved.end()) {
            _refs_being_resolved.insert(ref);
            json resolved = _refs[ref];
            ref_name = visit(resolved, ref_name);
            _refs_being_resolved.erase(ref);
        }
        return ref_name;
    }

    string _build_object_rule(
        const vector<pair<string, json>>& properties,
        const unordered_set<string>& required,
        const string& name,
        const json& additional_properties)
    {
        vector<string> required_props;
        vector<string> optional_props;
        unordered_map<string, string> prop_kv_rule_names;
        for (const auto& kv : properties) {
            const auto &prop_name = kv.first;
            const auto &prop_schema = kv.second;

            string prop_rule_name = visit(prop_schema, name + (name.empty() ? "" : "-") + prop_name);
            prop_kv_rule_names[prop_name] = _add_rule(
                name + (name.empty() ? "" : "-") + prop_name + "-kv",
                _format_literal(prop_name) + " space \":\" space " + prop_rule_name
            );
            if (required.find(prop_name) != required.end()) {
                required_props.push_back(prop_name);
            } else {
                optional_props.push_back(prop_name);
            }
        }
        if (additional_properties.is_object() || (additional_properties.is_boolean() && additional_properties.get<bool>())) {
            string sub_name = name + (name.empty() ? "" : "-") + "additional";
            string value_rule = visit(additional_properties.is_object() ? additional_properties : json::object(), sub_name + "-value");
            string kv_rule = _add_rule(sub_name + "-kv", _add_rule("string", PRIMITIVE_RULES.at("string")) + " \":\" space " + value_rule);
            prop_kv_rule_names["*"] = kv_rule;
            optional_props.push_back("*");
        }

        string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); i++) {
            if (i > 0) {
                rule += " \",\" space ";
            }
            rule += prop_kv_rule_names[required_props[i]];
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) {
                rule += " \",\" space ( ";
            }

            function<string(const vector<string>&, bool)> get_recursive_refs = [&](const vector<string>& ks, bool first_is_optional) {
                string res;
                if (ks.empty()) {
                    return res;
                }
                string k = ks[0];
                string kv_rule_name = prop_kv_rule_names[k];
                if (k == "*") {
                    res = _add_rule(
                        name + (name.empty() ? "" : "-") + "additional-kvs",
                        kv_rule_name + " ( \",\" space " + kv_rule_name + " )*"
                    );
                } else if (first_is_optional) {
                    res = "( \",\" space " + kv_rule_name + " )?";
                } else {
                    res = kv_rule_name;
                }
                if (ks.size() > 1) {
                    res += " " + _add_rule(
                        name + (name.empty() ? "" : "-") + k + "-rest",
                        get_recursive_refs(vector<string>(ks.begin() + 1, ks.end()), true)
                    );
                }
                return res;
            };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) {
                    rule += " | ";
                }
                rule += get_recursive_refs(vector<string>(optional_props.begin() + i, optional_props.end()), false);
            }
            if (!required_props.empty()) {
                rule += " )";
            }
            rule += " )?";
        }

        rule += " \"}\" space";

        return rule;
    }

public:
    SchemaConverter(
        const std::function<json(const string&)>& fetch_json,
        bool dotall)
          : _fetch_json(fetch_json), _dotall(dotall)
    {
        _rules["space"] = SPACE_RULE;
    }

    void resolve_refs(nlohmann::json& schema, const std::string& url) {
        /*
        * Resolves all $ref fields in the given schema, fetching any remote schemas,
        * replacing each $ref with absolute reference URL and populates _refs with the
        * respective referenced (sub)schema dictionaries.
        */
        function<void(json&)> visit_refs = [&](json& n) {
            if (n.is_array()) {
                for (auto& x : n) {
                    visit_refs(x);
                }
            } else if (n.is_object()) {
                if (n.contains("$ref")) {
                    string ref = n["$ref"];
                    if (_refs.find(ref) == _refs.end()) {
                        json target;
                        if (ref.find("https://") == 0) {
                            string base_url = ref.substr(0, ref.find('#'));
                            auto it = _refs.find(base_url);
                            if (it != _refs.end()) {
                                target = it->second;
                            } else {
                                // Fetch the referenced schema and resolve its refs
                                auto referenced = _fetch_json(ref);
                                resolve_refs(referenced, base_url);
                                _refs[base_url] = referenced;
                            }
                            if (ref.find('#') == string::npos || ref.substr(ref.find('#') + 1).empty()) {
                                return;
                            }
                        } else if (ref.find("#/") == 0) {
                            target = schema;
                            n["$ref"] = url + ref;
                            ref = url + ref;
                        } else {
                            _errors.push_back("Unsupported ref: " + ref);
                            return;
                        }
                        string pointer = ref.substr(ref.find('#') + 1);
                        vector<string> tokens = split(pointer, "/");
                        for (size_t i = 1; i < tokens.size(); ++i) {
                            string sel = tokens[i];
                            if (target.is_null() || !target.contains(sel)) {
                                _errors.push_back("Error resolving ref " + ref + ": " + sel + " not in " + target.dump());
                                return;
                            }
                            target = target[sel];
                        }
                        _refs[ref] = target;
                    }
                } else {
                    for (auto& kv : n.items()) {
                        visit_refs(kv.value());
                    }
                }
            }
        };

        visit_refs(schema);
    }

    string _generate_constant_rule(const json& value) {
        if (!value.is_string()) {
            _errors.push_back("Only string constants are supported, got " + value.dump());
            return "";
        }
        return _format_literal(value.get<string>());
    }

    string visit(const json& schema, const string& name) {
        json schema_type = schema.contains("type") ? schema["type"] : json();
        string schema_format = schema.contains("format") ? schema["format"].get<string>() : "";
        string rule_name = is_reserved_name(name) ? name + "-" : name.empty() ? "root" : name;

        if (schema.contains("$ref")) {
            return _add_rule(rule_name, _resolve_ref(schema["$ref"]));
        } else if (schema.contains("oneOf") || schema.contains("anyOf")) {
            vector<json> alt_schemas = schema.contains("oneOf") ? schema["oneOf"].get<vector<json>>() : schema["anyOf"].get<vector<json>>();
            return _add_rule(rule_name, _generate_union_rule(name, alt_schemas));
        } else if (schema_type.is_array()) {
            vector<json> schema_types;
            for (const auto& t : schema_type) {
                schema_types.push_back({{"type", t}});
            }
            return _add_rule(rule_name, _generate_union_rule(name, schema_types));
        } else if (schema.contains("const")) {
            return _add_rule(rule_name, _generate_constant_rule(schema["const"]));
        } else if (schema.contains("enum")) {
            vector<string> enum_values;
            for (const auto& v : schema["enum"]) {
                enum_values.push_back(_generate_constant_rule(v));
            }
            return _add_rule(rule_name, join(enum_values.begin(), enum_values.end(), " | "));
        } else if ((schema_type.is_null() || schema_type == "object")
                && (schema.contains("properties") ||
                    (schema.contains("additionalProperties") && schema["additionalProperties"] != true))) {
            unordered_set<string> required;
            if (schema.contains("required") && schema["required"].is_array()) {
                for (const auto& item : schema["required"]) {
                    if (item.is_string()) {
                        required.insert(item.get<string>());
                    }
                }
            }
            vector<pair<string, json>> properties;
            if (schema.contains("properties")) {
                for (const auto& prop : schema["properties"].items()) {
                    properties.emplace_back(prop.key(), prop.value());
                }
            }
            return _add_rule(rule_name,
                _build_object_rule(
                    properties, required, name,
                    schema.contains("additionalProperties") ? schema["additionalProperties"] : json()));
        } else if ((schema_type.is_null() || schema_type == "object") && schema.contains("allOf")) {
            unordered_set<string> required;
            vector<pair<string, json>> properties;
            string hybrid_name = name;
            std::function<void(const json&, bool)> add_component = [&](const json& comp_schema, bool is_required) {
                if (comp_schema.contains("$ref")) {
                    add_component(_refs[comp_schema["$ref"]], is_required);
                } else if (comp_schema.contains("properties")) {
                    for (const auto& prop : comp_schema["properties"].items()) {
                        properties.emplace_back(prop.key(), prop.value());
                        if (is_required) {
                            required.insert(prop.key());
                        }
                    }
                } else {
                  // todo warning
                }
            };
            for (auto& t : schema["allOf"]) {
                if (t.contains("anyOf")) {
                    for (auto& tt : t["anyOf"]) {
                        add_component(tt, false);
                    }
                } else {
                    add_component(t, true);
                }
            }
            return _add_rule(rule_name, _build_object_rule(properties, required, hybrid_name, json()));
        } else if ((schema_type.is_null() || schema_type == "array") && (schema.contains("items") || schema.contains("prefixItems"))) {
            json items = schema.contains("items") ? schema["items"] : schema["prefixItems"];
            if (items.is_array()) {
                string rule = "\"[\" space ";
                for (size_t i = 0; i < items.size(); i++) {
                    if (i > 0) {
                        rule += " \",\" space ";
                    }
                    rule += visit(items[i], name + (name.empty() ? "" : "-") + "tuple-" + to_string(i));
                }
                rule += " \"]\" space";
                return _add_rule(rule_name, rule);
            } else {
                string item_rule_name = visit(items, name + (name.empty() ? "" : "-") + "item");
                string list_item_operator = "( \",\" space " + item_rule_name + " )";
                string successive_items;
                int min_items = schema.contains("minItems") ? schema["minItems"].get<int>() : 0;
                json max_items_json = schema.contains("maxItems") ? schema["maxItems"] : json();
                int max_items = max_items_json.is_number_integer() ? max_items_json.get<int>() : -1;
                if (min_items > 0) {
                    successive_items += repeat(list_item_operator, min_items - 1);
                    min_items--;
                }
                if (max_items >= 0 && max_items > min_items) {
                    successive_items += repeat(list_item_operator + "?", max_items - min_items - 1);
                } else {
                    successive_items += list_item_operator + "*";
                }
                string rule;
                if (min_items == 0) {
                    rule =  "\"[\" space ( " + item_rule_name + " " + successive_items + " )? \"]\" space";
                } else {
                    rule =  "\"[\" space " + item_rule_name + " " + successive_items + " \"]\" space";
                }
                return _add_rule(rule_name, rule);
            }
        } else if ((schema_type.is_null() || schema_type == "string") && schema.contains("pattern")) {
            return _visit_pattern(schema["pattern"], rule_name);
        } else if ((schema_type.is_null() || schema_type == "string") && regex_match(schema_format, regex("^uuid[1-5]?$"))) {
            return _add_rule(rule_name == "root" ? "root" : schema_format, PRIMITIVE_RULES.at("uuid"));
        } else if ((schema_type.is_null() || schema_type == "string") && DATE_RULES.find(schema_format) != DATE_RULES.end()) {
            for (const auto& kv : DATE_RULES) {
                _add_rule(kv.first, kv.second);
            }
            return schema_format + "-string";
        } else if (schema.empty() || schema_type == "object") {
            for (const auto& n : OBJECT_RULE_NAMES) {
                _add_rule(n, PRIMITIVE_RULES.at(n));
            }
            return _add_rule(rule_name, "object");
        } else {
            if (!schema_type.is_string() || PRIMITIVE_RULES.find(schema_type.get<string>()) == PRIMITIVE_RULES.end()) {
                _errors.push_back("Unrecognized schema: " + schema.dump());
                return "";
            }
            // TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
            return _add_rule(rule_name == "root" ? "root" : schema_type.get<string>(), PRIMITIVE_RULES.at(schema_type.get<string>()));
        }
    }

    void check_errors() {
        if (!_errors.empty()) {
            throw std::runtime_error("JSON schema conversion failed:\n" + join(_errors.begin(), _errors.end(), "\n"));
        }
        if (!_warnings.empty()) {
            std::cerr << "WARNING: JSON schema conversion was incomplete: " + join(_warnings.begin(), _warnings.end(), "; ") << std::endl;
        }
    }

    string format_grammar() {
        stringstream ss;
        for (const auto& kv : _rules) {
            ss << kv.first << " ::= " << kv.second << endl;
        }
        return ss.str();
    }
};

string json_schema_to_grammar(const json& schema) {
    SchemaConverter converter([](const string&) { return json::object(); }, /* dotall= */ false);
    auto copy = schema;
    converter.resolve_refs(copy, "input");
    converter.visit(copy, "");
    converter.check_errors();
    return converter.format_grammar();
}
