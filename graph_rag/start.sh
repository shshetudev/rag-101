#!/bin/bash

# Graph RAG Startup Script

echo "=================================="
echo "   Graph RAG Project Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if .env exists
if [ ! -f .env ]; then
    print_info "Creating .env file from .env.example..."
    cp .env.example .env
    print_success ".env file created"
else
    print_success ".env file already exists"
fi

# Ask user what they want to do
echo ""
echo "What would you like to do?"
echo "1) Start both FastAPI and Neo4j (Full stack)"
echo "2) Start only Neo4j database"
echo "3) Run locally (install dependencies)"
echo "4) Stop all services"
echo "5) View logs"
echo "6) Run tests"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        print_info "Starting FastAPI and Neo4j services..."
        docker-compose up -d
        print_success "Services started!"
        print_info "API available at: http://localhost:8000"
        print_info "API docs at: http://localhost:8000/docs"
        print_info "Neo4j browser at: http://localhost:7474"
        ;;
    2)
        print_info "Starting Neo4j database only..."
        docker-compose -f docker-compose-neo4j.yml up -d
        print_success "Neo4j started!"
        print_info "Neo4j browser at: http://localhost:7474"
        print_info "Username: neo4j, Password: password123"
        ;;
    3)
        print_info "Setting up local development environment..."
        
        # Check if virtual environment exists
        if [ ! -d "venv" ]; then
            print_info "Creating virtual environment..."
            python3 -m venv venv
            print_success "Virtual environment created"
        fi
        
        print_info "Activating virtual environment..."
        source venv/bin/activate
        
        print_info "Installing dependencies..."
        pip install -r requirements.txt
        
        print_info "Downloading spaCy model..."
        python -m spacy download en_core_web_sm
        
        print_success "Setup complete!"
        print_info "To start the application, run:"
        print_info "  source venv/bin/activate"
        print_info "  uvicorn src.main:app --reload"
        ;;
    4)
        print_info "Stopping all services..."
        docker-compose down
        docker-compose -f docker-compose-neo4j.yml down
        print_success "All services stopped"
        ;;
    5)
        print_info "Showing logs (press Ctrl+C to exit)..."
        docker-compose logs -f
        ;;
    6)
        print_info "Running tests..."
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        pytest -v
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_success "Done!"
