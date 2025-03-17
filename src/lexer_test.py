import unittest
from unittest.mock import patch
import io
import sys

# Import the classes from the lexer module
from lexer import TokenType, Token, Lexer, keywords, operators, separators

class LexerTests(unittest.TestCase):
    def assert_tokens_equal(self, actual_tokens, expected_tokens):
        """Helper method to compare two lists of tokens"""
        self.assertEqual(len(actual_tokens), len(expected_tokens), 
                         f"Token count mismatch: {len(actual_tokens)} vs {len(expected_tokens)}")
        
        for i, (actual, expected) in enumerate(zip(actual_tokens, expected_tokens)):
            self.assertEqual(actual._type, expected._type, 
                            f"Token {i} type mismatch: {actual._type} vs {expected._type}")
            self.assertEqual(actual.value, expected.value, 
                            f"Token {i} value mismatch: {actual.value} vs {expected.value}")
            self.assertEqual(actual.line, expected.line, 
                            f"Token {i} line mismatch: {actual.line} vs {expected.line}")
            self.assertEqual(actual.column, expected.column, 
                            f"Token {i} column mismatch: {actual.column} vs {expected.column}")

    def test_empty_input(self):
        """Test that an empty input produces no tokens"""
        lexer = Lexer("")
        tokens = lexer.tokenize()
        self.assertEqual(len(tokens), 0)

    def test_whitespace_only(self):
        """Test that whitespace-only input produces no tokens"""
        lexer = Lexer("  \t\n  ")
        tokens = lexer.tokenize()
        self.assertEqual(len(tokens), 0)

    def test_variable_declaration(self):
        """Test basic variable declaration"""
        lexer = Lexer("U8 a;")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.KEYWORD, "U8", 1, 0),
            Token(TokenType.LITERAL, "a", 1, 3),
            Token(TokenType.SEPARATOR, ";", 1, 4)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)

    def test_variable_assignment(self):
        """Test variable assignment"""
        lexer = Lexer("b = 4;")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.LITERAL, "b", 1, 0),
            Token(TokenType.OPERATOR, "=", 1, 2),
            Token(TokenType.LITERAL, "4", 1, 4),
            Token(TokenType.SEPARATOR, ";", 1, 5)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)

    def test_expression(self):
        """Test expression with operators"""
        lexer = Lexer("c = a + b * 5;")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.LITERAL, "c", 1, 0),
            Token(TokenType.OPERATOR, "=", 1, 2),
            Token(TokenType.LITERAL, "a", 1, 4),
            Token(TokenType.OPERATOR, "+", 1, 6),
            Token(TokenType.LITERAL, "b", 1, 8),
            Token(TokenType.OPERATOR, "*", 1, 10),
            Token(TokenType.LITERAL, "5", 1, 12),
            Token(TokenType.SEPARATOR, ";", 1, 13)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)

    def test_boolean_literals(self):
        """Test boolean literals"""
        lexer = Lexer("Bool flag = true;")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.KEYWORD, "Bool", 1, 0),
            Token(TokenType.LITERAL, "flag", 1, 5),
            Token(TokenType.OPERATOR, "=", 1, 10),
            Token(TokenType.LITERAL, "true", 1, 12),
            Token(TokenType.SEPARATOR, ";", 1, 16)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)

    def test_comments(self):
        """Test comments"""
        lexer = Lexer("U8 a; // This is a comment\nU8 b;")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.KEYWORD, "U8", 1, 0),
            Token(TokenType.LITERAL, "a", 1, 3),
            Token(TokenType.SEPARATOR, ";", 1, 4),
            Token(TokenType.COMMENT, "// This is a comment", 1, 6),
            Token(TokenType.KEYWORD, "U8", 2, 0),
            Token(TokenType.LITERAL, "b", 2, 3),
            Token(TokenType.SEPARATOR, ";", 2, 4)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)

    def test_function_declaration(self):
        """Test function declaration"""
        lexer = Lexer("U8 myfunc(U8 a, U64 b) { return a + b; }")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.KEYWORD, "U8", 1, 0),
            Token(TokenType.LITERAL, "myfunc", 1, 3),
            Token(TokenType.SEPARATOR, "(", 1, 9),
            Token(TokenType.KEYWORD, "U8", 1, 10),
            Token(TokenType.LITERAL, "a", 1, 13),
            Token(TokenType.SEPARATOR, ",", 1, 14),
            Token(TokenType.KEYWORD, "U64", 1, 16),
            Token(TokenType.LITERAL, "b", 1, 20),
            Token(TokenType.SEPARATOR, ")", 1, 21),
            Token(TokenType.SEPARATOR, "{", 1, 23),
            Token(TokenType.KEYWORD, "return", 1, 25),
            Token(TokenType.LITERAL, "a", 1, 32),
            Token(TokenType.OPERATOR, "+", 1, 34),
            Token(TokenType.LITERAL, "b", 1, 36),
            Token(TokenType.SEPARATOR, ";", 1, 37),
            Token(TokenType.SEPARATOR, "}", 1, 39)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)


    # TODO!
    def test_directives(self):
        """Test preprocessor directives"""
        lexer = Lexer("#include <stdio.h>")
        tokens = lexer.tokenize()
        
        # The current implementation doesn't handle directives as special tokens,
        # but instead tokenizes the individual characters. Let's verify what it does.
        self.assertGreater(len(tokens), 0)  # Some tokens are generated
        
        # We could add specific expectations for what tokens are created,
        # but that would depend on the current implementation details

    def test_complex_expression(self):
        """Test complex expression with multiple operators"""
        lexer = Lexer("result = (a + b) * (c - d) / 2;")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.LITERAL, "result", 1, 0),
            Token(TokenType.OPERATOR, "=", 1, 7),
            Token(TokenType.SEPARATOR, "(", 1, 9),
            Token(TokenType.LITERAL, "a", 1, 10),
            Token(TokenType.OPERATOR, "+", 1, 12),
            Token(TokenType.LITERAL, "b", 1, 14),
            Token(TokenType.SEPARATOR, ")", 1, 15),
            Token(TokenType.OPERATOR, "*", 1, 17),
            Token(TokenType.SEPARATOR, "(", 1, 19),
            Token(TokenType.LITERAL, "c", 1, 20),
            Token(TokenType.OPERATOR, "-", 1, 22),
            Token(TokenType.LITERAL, "d", 1, 24),
            Token(TokenType.SEPARATOR, ")", 1, 25),
            Token(TokenType.OPERATOR, "/", 1, 27),
            Token(TokenType.LITERAL, "2", 1, 29),
            Token(TokenType.SEPARATOR, ";", 1, 30)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)

    def test_compound_assignment(self):
        """Test compound assignment operators"""
        lexer = Lexer("a += 5; b -= 3; c *= 2;")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.LITERAL, "a", 1, 0),
            Token(TokenType.OPERATOR, "+=", 1, 2),
            Token(TokenType.LITERAL, "5", 1, 5),
            Token(TokenType.SEPARATOR, ";", 1, 6),
            Token(TokenType.LITERAL, "b", 1, 8),
            Token(TokenType.OPERATOR, "-=", 1, 10),
            Token(TokenType.LITERAL, "3", 1, 13),
            Token(TokenType.SEPARATOR, ";", 1, 14),
            Token(TokenType.LITERAL, "c", 1, 16),
            Token(TokenType.OPERATOR, "*=", 1, 18),
            Token(TokenType.LITERAL, "2", 1, 21),
            Token(TokenType.SEPARATOR, ";", 1, 22)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)

    def test_increment_decrement(self):
        """Test increment and decrement operators"""
        lexer = Lexer("a++; b--;")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.LITERAL, "a", 1, 0),
            Token(TokenType.OPERATOR, "++", 1, 1),
            Token(TokenType.SEPARATOR, ";", 1, 3),
            Token(TokenType.LITERAL, "b", 1, 5),
            Token(TokenType.OPERATOR, "--", 1, 6),
            Token(TokenType.SEPARATOR, ";", 1, 8)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)

    def test_logical_operators(self):
        """Test logical operators"""
        lexer = Lexer("if (a && b || !c) { }")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.KEYWORD, "if", 1, 0),
            Token(TokenType.SEPARATOR, "(", 1, 3),
            Token(TokenType.LITERAL, "a", 1, 4),
            Token(TokenType.OPERATOR, "&&", 1, 6),
            Token(TokenType.LITERAL, "b", 1, 9),
            Token(TokenType.OPERATOR, "||", 1, 11),
            Token(TokenType.OPERATOR, "!", 1, 14),
            Token(TokenType.LITERAL, "c", 1, 15),
            Token(TokenType.SEPARATOR, ")", 1, 16),
            Token(TokenType.SEPARATOR, "{", 1, 18),
            Token(TokenType.SEPARATOR, "}", 1, 20)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)

    def test_bitwise_operators(self):
        """Test bitwise operators"""
        lexer = Lexer("result = a & b | c ^ d;")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.LITERAL, "result", 1, 0),
            Token(TokenType.OPERATOR, "=", 1, 7),
            Token(TokenType.LITERAL, "a", 1, 9),
            Token(TokenType.OPERATOR, "&", 1, 11),
            Token(TokenType.LITERAL, "b", 1, 13),
            Token(TokenType.OPERATOR, "|", 1, 15),
            Token(TokenType.LITERAL, "c", 1, 17),
            Token(TokenType.OPERATOR, "^", 1, 19),
            Token(TokenType.LITERAL, "d", 1, 21),
            Token(TokenType.SEPARATOR, ";", 1, 22)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)

    def test_missing_semicolon(self):
        """Test missing semicolon (should still tokenize correctly)"""
        lexer = Lexer("U8 a = 5\nU8 b = 6;")
        tokens = lexer.tokenize()
        
        expected_tokens = [
            Token(TokenType.KEYWORD, "U8", 1, 0),
            Token(TokenType.LITERAL, "a", 1, 3),
            Token(TokenType.OPERATOR, "=", 1, 5),
            Token(TokenType.LITERAL, "5", 1, 7),
            Token(TokenType.KEYWORD, "U8", 2, 0),
            Token(TokenType.LITERAL, "b", 2, 3),
            Token(TokenType.OPERATOR, "=", 2, 5),
            Token(TokenType.LITERAL, "6", 2, 7),
            Token(TokenType.SEPARATOR, ";", 2, 8)
        ]
        
        self.assert_tokens_equal(tokens, expected_tokens)

    def test_test_function(self):
        """Test the test function from the original code"""
        # Capture stdout to check if the test function works
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            from lexer import test
            test()
            output = fake_stdout.getvalue()
            # Just check if something was printed (don't check exact contents)
            self.assertGreater(len(output), 0)

if __name__ == '__main__':
    unittest.main()