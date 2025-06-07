'use client';

import React from 'react';

interface Doc {
  doc: string;
  similarity: number;
}

export default function Home() {
  const [searchTerm, setSearchTerm] = React.useState('');
  const [searchedTerm, setSearchedTerm] = React.useState('');
  const [results, setResults] = React.useState<Doc[]>([]);
  const [isLoading, setIsLoading] = React.useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    
    setResults([]);
    setIsLoading(true);

    const response = await fetch('http://localhost:7000/semantic-search/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: searchTerm }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    setResults(data.results || []);
    setSearchedTerm(data.query || '');
    setIsLoading(false);
  }

  return (
    <div className="w-full min-h-screen flex flex-col items-center py-20">
      <div className='flex flex-col gap-4'>
        <form onSubmit={handleSearch}>
          <div className='flex items-center gap-4'>
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Digite sua busca..."
              className='p-4 w-[600px] bg-gray-100/50 border border-gray-300 rounded-lg '
            />
            <button type="submit" className='cursor-pointer font-semibold transition-colors'>Buscar</button>
          </div>
        </form>

        {isLoading && (
          <div className='mt-4 text-gray-600'>Carregando...</div>
        )}

        {results.length > 0 ? (
          <div>
            <h2 className='text-gray-600 font-medium'>Resultados para: <span className='text-gray-800 font-semibold'>{searchedTerm}</span></h2>
            <ul className='mt-4 space-y-6'>
              {results.map((result, index) => (
                <li key={index} className=''>
                  <p className='text-sm text-gray-600'>Similaridade: {result.similarity.toFixed(5)}</p>
                  <div className='p-4 bg-white border border-gray-200 rounded-lg shadow-sm'>
                    <p className='text-gray-800 font-medium'>{result.doc}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>

    </div>
  );
}
