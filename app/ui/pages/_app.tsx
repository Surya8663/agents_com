import Head from 'next/head';
import Script from 'next/script';  // IMPORTANT: Add this import
import type { AppProps } from 'next/app'
import { Toaster } from 'react-hot-toast'

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        {/* Title and meta tags go here, but NO <script> tags */}
        <title>Document Intelligence System</title>
        <meta name="description" content="Multi-modal document intelligence with RAG" />
      </Head>
      
      {/* Tailwind CSS via CDN using Script component */}
      <Script 
        src="https://cdn.tailwindcss.com" 
        strategy="afterInteractive" 
      />
      
      <Component {...pageProps} />
      <Toaster position="top-right" />
    </>
  )
}