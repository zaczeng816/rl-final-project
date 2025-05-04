/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Disable automatic HTTPS redirect
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Forwarded-Proto',
            value: 'https',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig; 