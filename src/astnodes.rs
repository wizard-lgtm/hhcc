use std::fmt;

// Enum for node types for expressions
#[derive(Debug, Clone, PartialEq)]
pub enum ExpressionType {
    Literal,
    BinaryOp,
    UnaryOp,
    Assignment,
    Call,
    ArrayAccess,
    StructAccess,
    Reference,
}

// Enum for AST node types
#[derive(Debug, Clone, PartialEq)]
pub enum ASTNodeType {
    // Expression nodes
    Expression,
    
    // Block structure
    Block,
    
    // Variable operations
    VariableDeclaration,
    VariableAssignment,
    
    // Control flow
    Return,
    IfStatement,
    ElseStatement,
    WhileLoop,
    ForLoop,
    Break,
    Continue,
    
    // Function related
    FunctionDefinition,
    FunctionCall,
    
    // Classes and custom types
    ClassDefinition,
    UnionDefinition,
    
    // Arrays
    ArrayDeclaration,
    ArrayInitialization,
    
    // Pointers and references
    Pointer,
    Reference,
    
    // Metadata
    Comment,
}

// Variable structure
#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub value: Option<Box<dyn ASTNode>>,
}

impl Variable {
    pub fn new(name: String, value: Option<Box<dyn ASTNode>>) -> Self {
        Self { name, value }
    }
}

// ASTNode trait for common functionality
pub trait ASTNode: fmt::Debug {
    fn print_tree(&self, prefix: &str) -> String;
    fn node_type(&self) -> ASTNodeType;
    fn clone_box(&self) -> Box<dyn ASTNode>;
}

// Allow nodes to be cloned
impl Clone for Box<dyn ASTNode> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// Expression Node
#[derive(Debug, Clone)]
pub struct ExpressionNode {
    pub expression_type: ExpressionType,
    pub value: Option<String>,
    pub left: Option<Box<dyn ASTNode>>,
    pub right: Option<Box<dyn ASTNode>>,
    pub op: Option<String>,
}

impl ExpressionNode {
    pub fn new(
        expression_type: ExpressionType,
        value: Option<String>,
        left: Option<Box<dyn ASTNode>>,
        right: Option<Box<dyn ASTNode>>,
        op: Option<String>,
    ) -> Self {
        Self {
            expression_type,
            value,
            left,
            right,
            op,
        }
    }
}

impl ASTNode for ExpressionNode {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}ExpressionNode ({:?}", prefix, self.expression_type);
        
        if let Some(op) = &self.op {
            result.push_str(&format!(": {}", op));
        }
        
        if let Some(value) = &self.value {
            result.push_str(&format!(": {}", value));
        }
        
        result.push_str(")\n");

        if let Some(left) = &self.left {
            result.push_str(&format!("{}├── left: {}", prefix, left.print_tree(&format!("{}│   ", prefix))));
        }
        
        if let Some(right) = &self.right {
            result.push_str(&format!("{}└── right: {}", prefix, right.print_tree(&format!("{}    ", prefix))));
        }

        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::Expression
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// Block
#[derive(Debug, Clone)]
pub struct Block {
    pub nodes: Vec<Box<dyn ASTNode>>,
}

impl Block {
    pub fn new(nodes: Vec<Box<dyn ASTNode>>) -> Self {
        Self { nodes }
    }
}

impl ASTNode for Block {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}Block\n", prefix);
        
        for node in &self.nodes {
            result.push_str(&node.print_tree(&format!("{}    ", prefix)));
        }
        
        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::Block
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// VariableDeclaration
#[derive(Debug, Clone)]
pub struct VariableDeclaration {
    pub var_type: String,
    pub name: String,
    pub value: Option<Box<dyn ASTNode>>,
    pub is_user_typed: bool,
    pub is_pointer: bool,
}

impl VariableDeclaration {
    pub fn new(
        var_type: String,
        name: String,
        value: Option<Box<dyn ASTNode>>,
        is_user_typed: bool,
        is_pointer: bool,
    ) -> Self {
        Self {
            var_type,
            name,
            value,
            is_user_typed,
            is_pointer,
        }
    }
}

impl ASTNode for VariableDeclaration {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}VariableDeclaration\n", prefix);
        result.push_str(&format!("{}├── var_type: {}\n", prefix, self.var_type));
        result.push_str(&format!("{}├── name: {}\n", prefix, self.name));
        
        if let Some(value) = &self.value {
            result.push_str(&format!("{}└── value: {}", prefix, value.print_tree(&format!("{}    ", prefix))));
        }
        
        if self.is_user_typed {
            result.push_str(&format!("{}└── user_typed: {}\n", prefix, self.is_user_typed));
        }
        
        if self.is_pointer {
            result.push_str(&format!("{}└── is_pointer: {}\n", prefix, self.is_pointer));
        }
        
        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::VariableDeclaration
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// VariableAssignment
#[derive(Debug, Clone)]
pub struct VariableAssignment {
    pub name: String,
    pub value: Option<Box<dyn ASTNode>>,
}

impl VariableAssignment {
    pub fn new(name: String, value: Option<Box<dyn ASTNode>>) -> Self {
        Self { name, value }
    }
}

impl ASTNode for VariableAssignment {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}VariableAssignment\n", prefix);
        result.push_str(&format!("{}├── name: {}\n", prefix, self.name));
        
        if let Some(value) = &self.value {
            result.push_str(&format!("{}└── value: {}", prefix, value.print_tree(&format!("{}    ", prefix))));
        }
        
        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::VariableAssignment
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// Return
#[derive(Debug, Clone)]
pub struct Return {
    pub expression: Box<dyn ASTNode>,
}

impl Return {
    pub fn new(expression: Box<dyn ASTNode>) -> Self {
        Self { expression }
    }
}

impl ASTNode for Return {
    fn print_tree(&self, prefix: &str) -> String {
        format!(
            "{}Return\n{}└── expression: {}\n",
            prefix,
            prefix,
            self.expression.print_tree(&format!("{}    ", prefix))
        )
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::Return
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// FunctionDefinition
#[derive(Debug, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub return_type: String,
    pub parameters: Vec<Box<dyn ASTNode>>,
    pub body: Box<dyn ASTNode>,
}

impl FunctionDefinition {
    pub fn new(
        name: String,
        return_type: String,
        parameters: Vec<Box<dyn ASTNode>>,
        body: Box<dyn ASTNode>,
    ) -> Self {
        Self {
            name,
            return_type,
            parameters,
            body,
        }
    }
}

impl ASTNode for FunctionDefinition {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}FunctionDefinition\n", prefix);
        result.push_str(&format!("{}├── name: {}\n", prefix, self.name));
        result.push_str(&format!("{}├── return_type: {}\n", prefix, self.return_type));
        
        if !self.parameters.is_empty() {
            result.push_str(&format!("{}├── parameters:\n", prefix));
            for param in &self.parameters {
                result.push_str(&param.print_tree(&format!("{}│  ", prefix)));
            }
        }
        
        result.push_str(&format!("{}└── body: {}", prefix, self.body.print_tree(&format!("{}    ", prefix))));
        
        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::FunctionDefinition
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// IfStatement
#[derive(Debug, Clone)]
pub struct IfStatement {
    pub condition: Box<dyn ASTNode>,
    pub if_body: Box<dyn ASTNode>,
    pub else_body: Option<Box<dyn ASTNode>>,
}

impl IfStatement {
    pub fn new(
        condition: Box<dyn ASTNode>,
        if_body: Box<dyn ASTNode>,
        else_body: Option<Box<dyn ASTNode>>,
    ) -> Self {
        Self {
            condition,
            if_body,
            else_body,
        }
    }
}

impl ASTNode for IfStatement {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}IfStatement\n", prefix);
        result.push_str(&format!(
            "{}├── condition: {}",
            prefix,
            self.condition.print_tree(&format!("{}│   ", prefix))
        ));
        
        result.push_str(&format!(
            "{}├── if_body: {}",
            prefix,
            self.if_body.print_tree(&format!("{}│   ", prefix))
        ));

        if let Some(else_body) = &self.else_body {
            result.push_str(&format!(
                "{}└── else_body: {}",
                prefix,
                else_body.print_tree(&format!("{}    ", prefix))
            ));
        }

        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::IfStatement
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// WhileLoop
#[derive(Debug, Clone)]
pub struct WhileLoop {
    pub condition: Box<dyn ASTNode>,
    pub body: Box<dyn ASTNode>,
}

impl WhileLoop {
    pub fn new(condition: Box<dyn ASTNode>, body: Box<dyn ASTNode>) -> Self {
        Self { condition, body }
    }
}

impl ASTNode for WhileLoop {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}WhileLoop\n", prefix);
        result.push_str(&format!(
            "{}├── condition: {}",
            prefix,
            self.condition.print_tree(&format!("{}│   ", prefix))
        ));
        
        result.push_str(&format!(
            "{}└── body: {}",
            prefix,
            self.body.print_tree(&format!("{}    ", prefix))
        ));
        
        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::WhileLoop
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// ForLoop
#[derive(Debug, Clone)]
pub struct ForLoop {
    pub initialization: Option<Box<dyn ASTNode>>,
    pub condition: Option<Box<dyn ASTNode>>,
    pub update: Option<Box<dyn ASTNode>>,
    pub body: Box<dyn ASTNode>,
}

impl ForLoop {
    pub fn new(
        initialization: Option<Box<dyn ASTNode>>,
        condition: Option<Box<dyn ASTNode>>,
        update: Option<Box<dyn ASTNode>>,
        body: Box<dyn ASTNode>,
    ) -> Self {
        Self {
            initialization,
            condition,
            update,
            body,
        }
    }
}

impl ASTNode for ForLoop {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}ForLoop\n", prefix);
        
        match &self.initialization {
            Some(init) => result.push_str(&format!(
                "{}├── initialization: {}",
                prefix,
                init.print_tree(&format!("{}│   ", prefix))
            )),
            None => result.push_str(&format!("{}├── initialization: None\n", prefix)),
        }
        
        match &self.condition {
            Some(cond) => result.push_str(&format!(
                "{}├── condition: {}",
                prefix,
                cond.print_tree(&format!("{}│   ", prefix))
            )),
            None => result.push_str(&format!("{}├── condition: None\n", prefix)),
        }
        
        match &self.update {
            Some(upd) => result.push_str(&format!(
                "{}├── update: {}",
                prefix,
                upd.print_tree(&format!("{}│   ", prefix))
            )),
            None => result.push_str(&format!("{}├── update: None\n", prefix)),
        }
        
        result.push_str(&format!(
            "{}└── body: {}",
            prefix,
            self.body.print_tree(&format!("{}    ", prefix))
        ));
        
        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::ForLoop
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// Comment
#[derive(Debug, Clone)]
pub struct Comment {
    pub text: String,
    pub is_inline: bool,
}

impl Comment {
    pub fn new(text: String, is_inline: bool) -> Self {
        Self { text, is_inline }
    }
}

impl ASTNode for Comment {
    fn print_tree(&self, prefix: &str) -> String {
        let type_info = if self.is_inline { "Inline" } else { "Block" };
        format!("{}Comment ({}): {}\n", prefix, type_info, self.text)
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::Comment
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// FunctionCall
#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Vec<Box<dyn ASTNode>>,
    pub has_parentheses: bool,
}

impl FunctionCall {
    pub fn new(name: String, arguments: Vec<Box<dyn ASTNode>>, has_parentheses: bool) -> Self {
        Self {
            name,
            arguments,
            has_parentheses,
        }
    }
}

impl ASTNode for FunctionCall {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}FunctionCall\n", prefix);
        result.push_str(&format!("{}├── name: {}\n", prefix, self.name));
        result.push_str(&format!("{}├── has_parentheses: {}\n", prefix, self.has_parentheses));
        
        if !self.arguments.is_empty() {
            result.push_str(&format!("{}└── arguments:\n", prefix));
            
            for (i, arg) in self.arguments.iter().enumerate() {
                if i < self.arguments.len() - 1 {
                    result.push_str(&format!(
                        "{}    ├── {}",
                        prefix,
                        arg.print_tree(&format!("{}    │   ", prefix))
                    ));
                } else {
                    result.push_str(&format!(
                        "{}    └── {}",
                        prefix,
                        arg.print_tree(&format!("{}        ", prefix))
                    ));
                }
            }
        } else {
            result.push_str(&format!("{}└── arguments: []\n", prefix));
        }
        
        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::FunctionCall
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// Class
#[derive(Debug, Clone)]
pub struct Class {
    pub name: String,
    pub fields: Vec<Box<dyn ASTNode>>,
    pub parent: Option<String>,
}

impl Class {
    pub fn new(name: String, fields: Vec<Box<dyn ASTNode>>, parent: Option<String>) -> Self {
        // In Rust we won't implement the dynamic type registration that was in Python
        // That would require a different approach like a global registry
        Self { name, fields, parent }
    }
}

impl ASTNode for Class {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}Class\n", prefix);
        result.push_str(&format!("{}├── name: {}\n", prefix, self.name));

        if let Some(parent) = &self.parent {
            result.push_str(&format!("{}├── parent: {}\n", prefix, parent));
        }

        if !self.fields.is_empty() {
            result.push_str(&format!("{}└── fields:\n", prefix));
            
            for (i, field) in self.fields.iter().enumerate() {
                if i < self.fields.len() - 1 {
                    result.push_str(&format!(
                        "{}    ├── {}",
                        prefix,
                        field.print_tree(&format!("{}    │   ", prefix))
                    ));
                } else {
                    result.push_str(&format!(
                        "{}    └── {}",
                        prefix,
                        field.print_tree(&format!("{}        ", prefix))
                    ));
                }
            }
        } else {
            result.push_str(&format!("{}└── fields: []\n", prefix));
        }

        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::ClassDefinition
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// Union
#[derive(Debug, Clone)]
pub struct Union {
    pub name: String,
    pub fields: Vec<Box<dyn ASTNode>>,
}

impl Union {
    pub fn new(name: String, fields: Vec<Box<dyn ASTNode>>) -> Self {
        // As with Class, we won't implement the dynamic type registration
        Self { name, fields }
    }
}

impl ASTNode for Union {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}Union\n", prefix);
        result.push_str(&format!("{}├── name: {}\n", prefix, self.name));

        if !self.fields.is_empty() {
            result.push_str(&format!("{}└── fields:\n", prefix));
            
            for (i, field) in self.fields.iter().enumerate() {
                if i < self.fields.len() - 1 {
                    result.push_str(&format!(
                        "{}    ├── {}",
                        prefix,
                        field.print_tree(&format!("{}    │   ", prefix))
                    ));
                } else {
                    result.push_str(&format!(
                        "{}    └── {}",
                        prefix,
                        field.print_tree(&format!("{}        ", prefix))
                    ));
                }
            }
        } else {
            result.push_str(&format!("{}└── fields: []\n", prefix));
        }

        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::UnionDefinition
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// Break
#[derive(Debug, Clone)]
pub struct Break;

impl Break {
    pub fn new() -> Self {
        Self
    }
}

impl ASTNode for Break {
    fn print_tree(&self, prefix: &str) -> String {
        format!("{}Break\n", prefix)
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::Break
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// Continue
#[derive(Debug, Clone)]
pub struct Continue;

impl Continue {
    pub fn new() -> Self {
        Self
    }
}

impl ASTNode for Continue {
    fn print_tree(&self, prefix: &str) -> String {
        format!("{}Continue\n", prefix)
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::Continue
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// ArrayDeclaration
#[derive(Debug, Clone)]
pub struct ArrayDeclaration {
    pub base_type: String,
    pub name: String,
    pub dimensions: Vec<Option<Box<dyn ASTNode>>>,
    pub initialization: Option<Box<dyn ASTNode>>,
}

impl ArrayDeclaration {
    pub fn new(
        base_type: String,
        name: String,
        dimensions: Vec<Option<Box<dyn ASTNode>>>,
        initialization: Option<Box<dyn ASTNode>>,
    ) -> Self {
        Self {
            base_type,
            name,
            dimensions,
            initialization,
        }
    }
}

impl ASTNode for ArrayDeclaration {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}ArrayDeclaration\n", prefix);
        result.push_str(&format!("{}├── base_type: {}\n", prefix, self.base_type));
        result.push_str(&format!("{}├── name: {}\n", prefix, self.name));
        
        result.push_str(&format!("{}├── dimensions:\n", prefix));
        for (i, dim) in self.dimensions.iter().enumerate() {
            if let Some(dim_node) = dim {
                if i < self.dimensions.len() - 1 {
                    result.push_str(&format!(
                        "{}│   ├── {}",
                        prefix,
                        dim_node.print_tree(&format!("{}│   │   ", prefix))
                    ));
                } else {
                    result.push_str(&format!(
                        "{}│   └── {}",
                        prefix,
                        dim_node.print_tree(&format!("{}│       ", prefix))
                    ));
                }
            } else {
                result.push_str(&format!("{}│   ├── dynamic[]\n", prefix));
            }
        }
        
        if let Some(init) = &self.initialization {
            result.push_str(&format!(
                "{}└── initialization: {}",
                prefix,
                init.print_tree(&format!("{}    ", prefix))
            ));
        }
        
        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::ArrayDeclaration
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// ArrayInitialization
#[derive(Debug, Clone)]
pub struct ArrayInitialization {
    pub elements: Vec<Box<dyn ASTNode>>,
}

impl ArrayInitialization {
    pub fn new(elements: Vec<Box<dyn ASTNode>>) -> Self {
        Self { elements }
    }
}

impl ASTNode for ArrayInitialization {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}ArrayInitialization\n", prefix);
        
        if !self.elements.is_empty() {
            for (i, element) in self.elements.iter().enumerate() {
                if i < self.elements.len() - 1 {
                    result.push_str(&format!(
                        "{}├── {}",
                        prefix,
                        element.print_tree(&format!("{}│   ", prefix))
                    ));
                } else {
                    result.push_str(&format!(
                        "{}└── {}",
                        prefix,
                        element.print_tree(&format!("{}    ", prefix))
                    ));
                }
            }
        } else {
            result.push_str(&format!("{}└── [empty]\n", prefix));
        }
        
        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::ArrayInitialization
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// Pointer
#[derive(Debug, Clone)]
pub struct Pointer {
    pub variable_name: String,
}

impl Pointer {
    pub fn new(variable_name: String) -> Self {
        Self { variable_name }
    }
}

impl ASTNode for Pointer {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}Pointer\n", prefix);
        result.push_str(&format!("{}└── variable_name: {}\n", prefix, self.variable_name));
        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::Pointer
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

// Reference
#[derive(Debug, Clone)]
pub struct Reference {
    pub variable_name: String,
}

impl Reference {
    pub fn new(variable_name: String) -> Self {
        Self { variable_name }
    }
}

impl ASTNode for Reference {
    fn print_tree(&self, prefix: &str) -> String {
        let mut result = format!("{}Reference\n", prefix);
        result.push_str(&format!("{}└── variable_name: {}\n", prefix, self.variable_name));
        result
    }

    fn node_type(&self) -> ASTNodeType {
        ASTNodeType::Reference
    }

    fn clone_box(&self) -> Box<dyn ASTNode> {
        Box::new(self.clone())
    }
}

